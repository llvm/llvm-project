#include <OffloadAPI.h>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

#define CHECK(expr)                                                            \
  do {                                                                         \
    ol_result_t result = (expr);                                               \
    if (result != OL_SUCCESS) {                                                \
      fprintf(stderr, "FAIL %s:%d: %s -> %d: %s\n", __FILE__, __LINE__, #expr, \
              result->Code, result->Details);                                  \
      return 1;                                                                \
    }                                                                          \
  } while (0)

static std::vector<uint8_t>
wrapOffloadBinary(const std::vector<uint8_t> &image) {
  static constexpr char key[] = "triple";
  static constexpr char triple[] = "spirv64-unknown-unknown";
  constexpr uint64_t headerSize = 40;
  constexpr uint64_t entrySize = 48;
  constexpr uint64_t stringEntrySize = 24;
  constexpr uint64_t keyOffset = headerSize + entrySize + stringEntrySize;
  constexpr uint64_t tripleOffset = keyOffset + sizeof(key);
  constexpr uint64_t imageOffset =
      (tripleOffset + sizeof(triple) - 1 + 7) & ~uint64_t{7};

  std::vector<uint8_t> blob(imageOffset + image.size(), 0);
  uint8_t *data = blob.data();
  data[0] = 0x10;
  data[1] = 0xFF;
  data[2] = 0x10;
  data[3] = 0xAD;
  auto put32 = [&](uint64_t offset, uint32_t value) {
    memcpy(data + offset, &value, sizeof(value));
  };
  auto put64 = [&](uint64_t offset, uint64_t value) {
    memcpy(data + offset, &value, sizeof(value));
  };
  put32(4, 2);
  put64(8, blob.size());
  put64(16, headerSize);
  put64(24, 1);
  put32(40, (uint32_t{8} << 16) | 1u);
  put64(48, headerSize + entrySize);
  put64(56, 1);
  put64(64, imageOffset);
  put64(72, image.size());
  put64(88, keyOffset);
  put64(96, tripleOffset);
  put64(104, sizeof(triple) - 1);
  memcpy(data + keyOffset, key, sizeof(key));
  memcpy(data + tripleOffset, triple, sizeof(triple) - 1);
  memcpy(data + imageOffset, image.data(), image.size());
  return blob;
}

static std::vector<uint8_t> readFile(const char *path) {
  FILE *file = fopen(path, "rb");
  if (!file) {
    perror(path);
    exit(1);
  }
  fseek(file, 0, SEEK_END);
  long size = ftell(file);
  fseek(file, 0, SEEK_SET);
  std::vector<uint8_t> contents(size);
  if (fread(contents.data(), 1, size, file) != static_cast<size_t>(size)) {
    perror("read");
    exit(1);
  }
  fclose(file);
  return contents;
}

static ol_device_handle_t pickDevice() {
  struct Search {
    const char *requiredName;
    ol_device_handle_t device = nullptr;
  } search{getenv("INTER_DEVICE_NAME")};
  olIterateDevices(
      [](ol_device_handle_t device, void *data) -> bool {
        auto *search = static_cast<Search *>(data);
        ol_device_type_t type;
        if (olGetDeviceInfo(device, OL_DEVICE_INFO_TYPE, sizeof(type), &type) !=
                OL_SUCCESS ||
            type != OL_DEVICE_TYPE_GPU)
          return true;
        ol_platform_handle_t platform;
        if (olGetDeviceInfo(device, OL_DEVICE_INFO_PLATFORM, sizeof(platform),
                            &platform) != OL_SUCCESS)
          return true;
        ol_platform_backend_t backend;
        if (olGetPlatformInfo(platform, OL_PLATFORM_INFO_BACKEND,
                              sizeof(backend), &backend) != OL_SUCCESS ||
            backend != OL_PLATFORM_BACKEND_LEVEL_ZERO)
          return true;
        char name[256] = {};
        if (olGetDeviceInfo(device, OL_DEVICE_INFO_NAME, sizeof(name), name) !=
            OL_SUCCESS)
          return true;
        if (search->requiredName && search->requiredName[0] != '\0' &&
            !strstr(name, search->requiredName))
          return true;
        fprintf(stderr, "device: %s\n", name);
        search->device = device;
        return false;
      },
      &search);
  return search.device;
}

int main(int argc, char **argv) {
  if (argc != 2 && argc != 4) {
    fprintf(stderr, "usage: %s <zebin.bin> [size reduction-size]\n", argv[0]);
    return 1;
  }

  const bool structuredRandom = argc == 4;
  int64_t m = 128;
  int64_t k = 64;
  if (structuredRandom) {
    auto parseInteger = [](const char *text, int64_t &value) {
      const char *end = text + strlen(text);
      const std::from_chars_result result = std::from_chars(text, end, value);
      return result.ec == std::errc{} && result.ptr == end;
    };
    if (!parseInteger(argv[2], m) || !parseInteger(argv[3], k)) {
      fprintf(stderr, "FAIL: size arguments must be integers\n");
      return 1;
    }
  }
  const int64_t n = m;
  if (m < 64 || m % 64 != 0 || k < 32 || k % 32 != 0) {
    fprintf(stderr,
            "FAIL: size must be a multiple of 64 and reduction size must be "
            "a multiple of 32\n");
    return 1;
  }
  const uint64_t maxSize = std::numeric_limits<size_t>::max();
  const uint64_t matrixSize = static_cast<uint64_t>(m);
  const uint64_t reductionSize = static_cast<uint64_t>(k);
  if (matrixSize > uint64_t{UINT32_MAX} * 64 ||
      matrixSize > maxSize / sizeof(float) / matrixSize ||
      matrixSize > maxSize / sizeof(_Float16) / reductionSize) {
    fprintf(stderr, "FAIL: matrix dimensions are too large\n");
    return 1;
  }
  const size_t matrixElements = matrixSize * matrixSize;
  const size_t operandElements = matrixSize * reductionSize;
  CHECK(olInit(nullptr));
  ol_device_handle_t device = pickDevice();
  if (!device) {
    fprintf(stderr, "FAIL: no matching Level Zero GPU device found\n");
    return 1;
  }

  std::vector<uint8_t> blob = wrapOffloadBinary(readFile(argv[1]));
  bool valid = false;
  CHECK(olIsValidBinary(device, blob.data(), blob.size(), &valid));
  if (!valid) {
    fprintf(stderr, "FAIL: invalid offload binary\n");
    return 1;
  }
  ol_program_handle_t program;
  CHECK(olCreateProgram(device, blob.data(), blob.size(), &program));
  ol_symbol_handle_t kernel;
  CHECK(olGetSymbol(program, "payload_kernel", OL_SYMBOL_KIND_KERNEL, &kernel));

  void *aStorage = nullptr;
  void *bStorage = nullptr;
  void *cStorage = nullptr;
  CHECK(olMemAlloc(device, OL_ALLOC_TYPE_MANAGED,
                   operandElements * sizeof(_Float16), &aStorage));
  CHECK(olMemAlloc(device, OL_ALLOC_TYPE_MANAGED,
                   operandElements * sizeof(_Float16), &bStorage));
  CHECK(olMemAlloc(device, OL_ALLOC_TYPE_MANAGED,
                   matrixElements * sizeof(float), &cStorage));
  auto *a = static_cast<_Float16 *>(aStorage);
  auto *b = static_cast<_Float16 *>(bStorage);
  auto *c = static_cast<float *>(cStorage);

  std::vector<float> reference(matrixElements, 0.0f);
  std::mt19937 random(0x4D41544Du);
  if (structuredRandom) {
    auto randomSign = [&]() { return (random() & 1) ? 1 : -1; };
    std::vector<int> rowSigns(m);
    std::vector<int> lhsSigns(k);
    std::vector<int> rhsSigns(k);
    std::vector<int> columnSigns(n);
    std::generate(rowSigns.begin(), rowSigns.end(), randomSign);
    std::generate(lhsSigns.begin(), lhsSigns.end(), randomSign);
    std::generate(rhsSigns.begin(), rhsSigns.end(), randomSign);
    std::generate(columnSigns.begin(), columnSigns.end(), randomSign);
    for (int64_t row = 0; row < m; ++row)
      for (int64_t inner = 0; inner < k; ++inner)
        a[row * k + inner] =
            static_cast<_Float16>(rowSigns[row] * lhsSigns[inner]);
    for (int64_t inner = 0; inner < k; ++inner)
      for (int64_t column = 0; column < n; ++column)
        b[inner * n + column] =
            static_cast<_Float16>(rhsSigns[inner] * columnSigns[column]);
    int64_t reduction = 0;
    for (int64_t inner = 0; inner < k; ++inner)
      reduction += lhsSigns[inner] * rhsSigns[inner];
    for (int64_t row = 0; row < m; ++row)
      for (int64_t column = 0; column < n; ++column)
        reference[row * n + column] =
            static_cast<float>(rowSigns[row] * columnSigns[column] * reduction);
  } else {
    std::uniform_int_distribution<int> distribution(-8, 8);
    for (int64_t index = 0; index < m * k; ++index)
      a[index] = static_cast<_Float16>(distribution(random) * 0.125f);
    for (int64_t index = 0; index < k * n; ++index)
      b[index] = static_cast<_Float16>(distribution(random) * 0.125f);
    for (int64_t row = 0; row < m; ++row)
      for (int64_t column = 0; column < n; ++column)
        for (int64_t inner = 0; inner < k; ++inner)
          reference[row * n + column] +=
              static_cast<float>(a[row * k + inner]) *
              static_cast<float>(b[inner * n + column]);
  }
  for (int64_t index = 0; index < m * n; ++index)
    c[index] = std::numeric_limits<float>::quiet_NaN();

  std::vector<void *> pointerValues = {aStorage, aStorage, bStorage,
                                       bStorage, cStorage, cStorage};
  std::vector<int64_t> scalars = {0, m, k, k, 1, 0, k, n, n, 1, 0, m, n, n, 1};
  std::vector<void *> arguments;
  std::vector<size_t> argumentSizes(21, sizeof(uint64_t));
  int pointerIndex = 0;
  int scalarIndex = 0;
  for (int descriptor = 0; descriptor < 3; ++descriptor) {
    arguments.push_back(&pointerValues[pointerIndex++]);
    arguments.push_back(&pointerValues[pointerIndex++]);
    for (int field = 0; field < 5; ++field)
      arguments.push_back(&scalars[scalarIndex++]);
  }

  ol_kernel_launch_size_args_t launch;
  launch.Dimensions = 3;
  launch.NumGroups = {static_cast<uint32_t>(m / 64),
                      static_cast<uint32_t>(n / 64), 1};
  launch.GroupSize = {256, 1, 1};
  launch.DynSharedMemory = 0;
  CHECK(olLaunchKernel(nullptr, device, kernel, &launch, nullptr,
                       arguments.size(), arguments.data(),
                       argumentSizes.data()));

  float maxError = 0.0f;
  for (int64_t index = 0; index < m * n; ++index) {
    float error = std::abs(c[index] - reference[index]);
    maxError = std::max(maxError, error);
    if (!std::isfinite(c[index]) || error != 0.0f) {
      fprintf(stderr, "FAIL: C[%ld,%ld] = %.9g, expected %.9g (error %.9g)\n",
              index / n, index % n, c[index], reference[index], error);
      return 1;
    }
  }
  printf("PASS: %ldx%ldx%ld %s f16 matmul, max error %.9g\n", m, n, k,
         structuredRandom ? "structured-random" : "random", maxError);

  CHECK(olMemFree(aStorage));
  CHECK(olMemFree(bStorage));
  CHECK(olMemFree(cStorage));
  CHECK(olDestroyProgram(program));
  CHECK(olShutDown());
  return 0;
}
