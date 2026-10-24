#include <level_zero/ze_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <utility>
#include <vector>

#define ZE_CHECK(expr)                                                         \
  do {                                                                         \
    ze_result_t result = (expr);                                               \
    if (result != ZE_RESULT_SUCCESS) {                                         \
      fprintf(stderr, "FAIL %s:%d: %s -> 0x%x\n", __FILE__, __LINE__, #expr,   \
              static_cast<unsigned>(result));                                  \
      return 1;                                                                \
    }                                                                          \
  } while (0)

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

static bool selectDevice(const char *requiredName, ze_driver_handle_t &driver,
                         ze_device_handle_t &device,
                         ze_device_properties_t &properties) {
  uint32_t driverCount = 0;
  if (zeDriverGet(&driverCount, nullptr) != ZE_RESULT_SUCCESS)
    return false;
  std::vector<ze_driver_handle_t> drivers(driverCount);
  if (zeDriverGet(&driverCount, drivers.data()) != ZE_RESULT_SUCCESS)
    return false;
  for (ze_driver_handle_t candidateDriver : drivers) {
    uint32_t deviceCount = 0;
    if (zeDeviceGet(candidateDriver, &deviceCount, nullptr) !=
        ZE_RESULT_SUCCESS)
      continue;
    std::vector<ze_device_handle_t> devices(deviceCount);
    if (zeDeviceGet(candidateDriver, &deviceCount, devices.data()) !=
        ZE_RESULT_SUCCESS)
      continue;
    for (ze_device_handle_t candidateDevice : devices) {
      ze_device_properties_t candidateProperties{
          ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES};
      if (zeDeviceGetProperties(candidateDevice, &candidateProperties) !=
              ZE_RESULT_SUCCESS ||
          !strstr(candidateProperties.name, requiredName))
        continue;
      driver = candidateDriver;
      device = candidateDevice;
      properties = candidateProperties;
      return true;
    }
  }
  return false;
}

int main(int argc, char **argv) {
  if ((argc != 10 && argc != 11) || (std::string(argv[1]) != "inter" &&
                                     std::string(argv[1]) != "lighthouse")) {
    fprintf(stderr,
            "usage: %s inter|lighthouse <zebin> <device> <warmups> <batches> "
            "<iterations> <size> <reduction-size> <kernel> [padding-k-tiles]\n",
            argv[0]);
    return 1;
  }
  const int warmups = std::atoi(argv[4]);
  const int batches = std::atoi(argv[5]);
  const int iterations = std::atoi(argv[6]);
  const int64_t size = std::atoll(argv[7]);
  const int64_t k = std::atoll(argv[8]);
  const int64_t paddingKTiles = argc == 11 ? std::atoll(argv[10]) : 0;
  if (warmups < 1 || batches < 1 || iterations < 1) {
    fprintf(stderr, "warmups, batches, and iterations must be positive\n");
    return 1;
  }
  if (size < 64 || size % 64 != 0) {
    fprintf(stderr, "matrix size must be a positive multiple of 64\n");
    return 1;
  }
  if (k < 32 || k % 32 != 0) {
    fprintf(stderr, "reduction size must be a positive multiple of 32\n");
    return 1;
  }
  if (paddingKTiles < 0) {
    fprintf(stderr, "padding K tiles must be nonnegative\n");
    return 1;
  }
  const int64_t m = size;
  const int64_t n = size;
  const int64_t operandElements = m * k;
  const int64_t paddedOperandElements =
      operandElements + paddingKTiles * 32 * m;
  constexpr int64_t outputGuardElements = 4096;
  const int64_t outputElements = m * n;
  const int64_t guardedOutputElements =
      outputElements + 2 * outputGuardElements;

  ZE_CHECK(zeInit(ZE_INIT_FLAG_GPU_ONLY));
  ze_driver_handle_t driver = nullptr;
  ze_device_handle_t device = nullptr;
  ze_device_properties_t properties{ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES};
  if (!selectDevice(argv[3], driver, device, properties)) {
    fprintf(stderr, "no Level Zero GPU matching '%s'\n", argv[3]);
    return 1;
  }

  ze_context_desc_t contextDesc{ZE_STRUCTURE_TYPE_CONTEXT_DESC};
  ze_context_handle_t context;
  ZE_CHECK(zeContextCreate(driver, &contextDesc, &context));
  ze_command_queue_desc_t queueDesc{ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC};
  queueDesc.mode = ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS;
  ze_command_list_handle_t commandList;
  ZE_CHECK(
      zeCommandListCreateImmediate(context, device, &queueDesc, &commandList));

  std::vector<uint8_t> binary = readFile(argv[2]);
  ze_module_desc_t moduleDesc{ZE_STRUCTURE_TYPE_MODULE_DESC};
  moduleDesc.format = ZE_MODULE_FORMAT_NATIVE;
  moduleDesc.inputSize = binary.size();
  moduleDesc.pInputModule = binary.data();
  ze_module_handle_t module;
  ZE_CHECK(zeModuleCreate(context, device, &moduleDesc, &module, nullptr));
  ze_kernel_desc_t kernelDesc{ZE_STRUCTURE_TYPE_KERNEL_DESC};
  kernelDesc.pKernelName = argv[9];
  ze_kernel_handle_t kernel;
  ZE_CHECK(zeKernelCreate(module, &kernelDesc, &kernel));
  ZE_CHECK(zeKernelSetGroupSize(kernel, 256, 1, 1));

  ze_device_mem_alloc_desc_t deviceAlloc{
      ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC};
  ze_host_mem_alloc_desc_t hostAlloc{ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC};
  void *aStorage = nullptr;
  void *bStorage = nullptr;
  void *cAllocation = nullptr;
  ZE_CHECK(zeMemAllocShared(context, &deviceAlloc, &hostAlloc,
                            paddedOperandElements * sizeof(_Float16), 64,
                            device, &aStorage));
  ZE_CHECK(zeMemAllocShared(context, &deviceAlloc, &hostAlloc,
                            paddedOperandElements * sizeof(_Float16), 64,
                            device, &bStorage));
  ZE_CHECK(zeMemAllocShared(context, &deviceAlloc, &hostAlloc,
                            guardedOutputElements * sizeof(float), 64, device,
                            &cAllocation));
  auto *a = static_cast<_Float16 *>(aStorage);
  auto *b = static_cast<_Float16 *>(bStorage);
  auto *cBase = static_cast<float *>(cAllocation);
  auto *c = cBase + outputGuardElements;
  void *cStorage = c;
  std::mt19937 random(0x4D41544Du);
  auto randomValue = [&]() { return static_cast<int>(random() % 3) - 1; };
  std::generate(a, a + operandElements,
                [&]() { return static_cast<_Float16>(randomValue()); });
  std::generate(b, b + operandElements,
                [&]() { return static_cast<_Float16>(randomValue()); });
  std::fill(a + operandElements, a + paddedOperandElements, _Float16{0});
  std::fill(b + operandElements, b + paddedOperandElements, _Float16{0});
  constexpr float prefixCanary = -12345.25f;
  constexpr float suffixCanary = 23456.5f;
  std::fill(cBase, c, prefixCanary);
  std::fill(c + outputElements, cBase + guardedOutputElements, suffixCanary);
  auto poisonOutput = [&]() {
    std::fill(c, c + outputElements, std::numeric_limits<float>::quiet_NaN());
  };
  poisonOutput();

  void *pointers[] = {aStorage, aStorage, bStorage,
                      bStorage, cStorage, cStorage};
  int64_t scalars[] = {0, m, k, k, 1, 0, k, n, n, 1, 0, m, n, n, 1};
  unsigned argument = 0;
  int pointerIndex = 0;
  int scalarIndex = 0;
  for (int descriptor = 0; descriptor < 3; ++descriptor) {
    ZE_CHECK(zeKernelSetArgumentValue(kernel, argument++, sizeof(void *),
                                      &pointers[pointerIndex++]));
    ZE_CHECK(zeKernelSetArgumentValue(kernel, argument++, sizeof(void *),
                                      &pointers[pointerIndex++]));
    for (int field = 0; field < 5; ++field)
      ZE_CHECK(zeKernelSetArgumentValue(kernel, argument++, sizeof(int64_t),
                                        &scalars[scalarIndex++]));
  }

  constexpr int64_t spotCheckCount = 1024;
  std::mt19937 spotRandom(0x53504F54u);
  std::vector<int64_t> spotIndices;
  spotIndices.reserve(std::min(spotCheckCount, outputElements));
  auto addSpot = [&](int64_t index) {
    if (std::find(spotIndices.begin(), spotIndices.end(), index) ==
        spotIndices.end())
      spotIndices.push_back(index);
  };
  addSpot(0);
  addSpot(n - 1);
  addSpot((m - 1) * n);
  addSpot(outputElements - 1);
  while (static_cast<int64_t>(spotIndices.size()) <
         std::min(spotCheckCount, outputElements))
    addSpot(static_cast<int64_t>(spotRandom()) % outputElements);

  std::vector<std::pair<int64_t, float>> spotReferences;
  spotReferences.reserve(spotIndices.size());
  for (int64_t index : spotIndices) {
    int64_t row = index / n;
    int64_t column = index % n;
    int64_t expected = 0;
    for (int64_t inner = 0; inner < k; ++inner)
      expected += static_cast<int>(a[row * k + inner]) *
                  static_cast<int>(b[inner * n + column]);
    spotReferences.emplace_back(index, static_cast<float>(expected));
  }

  ze_event_pool_desc_t poolDesc{ZE_STRUCTURE_TYPE_EVENT_POOL_DESC};
  poolDesc.flags =
      ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP | ZE_EVENT_POOL_FLAG_HOST_VISIBLE;
  poolDesc.count = 1;
  ze_event_pool_handle_t eventPool;
  ZE_CHECK(zeEventPoolCreate(context, &poolDesc, 1, &device, &eventPool));
  ze_event_desc_t eventDesc{ZE_STRUCTURE_TYPE_EVENT_DESC};
  eventDesc.signal = ZE_EVENT_SCOPE_FLAG_HOST;
  eventDesc.wait = ZE_EVENT_SCOPE_FLAG_HOST;
  ze_event_handle_t event;
  ZE_CHECK(zeEventCreate(eventPool, &eventDesc, &event));
  ze_group_count_t groups{static_cast<uint32_t>(size / 64),
                          static_cast<uint32_t>(size / 64), 1};
  auto launch = [&]() -> ze_result_t {
    ze_result_t result = zeCommandListAppendLaunchKernel(
        commandList, kernel, &groups, event, 0, nullptr);
    if (result != ZE_RESULT_SUCCESS)
      return result;
    return zeEventHostSynchronize(event, UINT64_MAX);
  };
  auto validateCanaries = [&]() {
    for (int64_t index = 0; index < outputGuardElements; ++index) {
      if (cBase[index] != prefixCanary) {
        fprintf(stderr, "corrupt C prefix canary at %ld: %.9g\n", index,
                cBase[index]);
        return false;
      }
      if (c[outputElements + index] != suffixCanary) {
        fprintf(stderr, "corrupt C suffix canary at %ld: %.9g\n", index,
                c[outputElements + index]);
        return false;
      }
    }
    return true;
  };
  auto validateSpots = [&]() {
    int64_t invalidCount = 0;
    int64_t minRow = m;
    int64_t maxRow = -1;
    int64_t minColumn = n;
    int64_t maxColumn = -1;
    for (auto [index, expected] : spotReferences)
      if (!std::isfinite(c[index]) || c[index] != expected) {
        int64_t row = index / n;
        int64_t column = index % n;
        if (invalidCount == 0)
          fprintf(stderr, "incorrect C[%ld,%ld]: %.9g != %.9g\n", row, column,
                  c[index], expected);
        ++invalidCount;
        minRow = std::min(minRow, row);
        maxRow = std::max(maxRow, row);
        minColumn = std::min(minColumn, column);
        maxColumn = std::max(maxColumn, column);
      }
    if (invalidCount == 0)
      return true;
    fprintf(
        stderr,
        "invalid spot checks: %ld, bounds rows %ld..%ld, columns %ld..%ld\n",
        invalidCount, minRow, maxRow, minColumn, maxColumn);
    return false;
  };
  auto validateFreivalds = [&](uint32_t seed) {
    constexpr int64_t prime = 65521;
    std::mt19937 projectionRandom(seed);
    std::vector<int64_t> projection(n);
    std::generate(projection.begin(), projection.end(), [&]() {
      return static_cast<int64_t>(projectionRandom() % prime);
    });
    std::vector<int64_t> projectedB(k);
    for (int64_t inner = 0; inner < k; ++inner) {
      int64_t sum = 0;
      for (int64_t column = 0; column < n; ++column)
        sum += static_cast<int>(b[inner * n + column]) * projection[column];
      projectedB[inner] = (sum % prime + prime) % prime;
    }
    for (int64_t row = 0; row < m; ++row) {
      int64_t expected = 0;
      int64_t observed = 0;
      for (int64_t inner = 0; inner < k; ++inner)
        expected += static_cast<int>(a[row * k + inner]) * projectedB[inner];
      for (int64_t column = 0; column < n; ++column)
        observed +=
            static_cast<int64_t>(c[row * n + column]) * projection[column];
      expected = (expected % prime + prime) % prime;
      observed = (observed % prime + prime) % prime;
      if (expected == observed)
        continue;
      fprintf(stderr, "Freivalds check 0x%x failed at row %ld: %ld != %ld\n",
              seed, row, observed, expected);
      return false;
    }
    return true;
  };
  auto validateFull = [&](bool algebraic) {
    if (!validateCanaries())
      return false;
    for (int64_t index = 0; index < outputElements; ++index) {
      float value = c[index];
      if (std::isfinite(value) && std::trunc(value) == value &&
          std::abs(value) <= k)
        continue;
      fprintf(stderr, "invalid C[%ld,%ld]: %.9g\n", index / n, index % n,
              value);
      return false;
    }
    if (!validateSpots())
      return false;
    if (!algebraic)
      return true;
    return validateFreivalds(0x46524531u) && validateFreivalds(0x46524532u);
  };

  for (int iteration = 0; iteration < warmups; ++iteration) {
    ZE_CHECK(launch());
    bool full = iteration == 0 || iteration + 1 == warmups;
    if (!(full ? validateFull(iteration == 0) : validateSpots()))
      return 1;
    ZE_CHECK(zeEventHostReset(event));
  }
  // Restore GPU residency after host validation before collecting timestamps.
  ZE_CHECK(launch());
  ZE_CHECK(zeEventHostReset(event));

  const uint64_t timestampMask =
      properties.kernelTimestampValidBits == 64
          ? ~uint64_t{0}
          : (uint64_t{1} << properties.kernelTimestampValidBits) - 1;
  std::vector<double> samples;
  for (int batch = 0; batch < batches; ++batch) {
    double elapsedNanoseconds = 0.0;
    for (int iteration = 0; iteration < iterations; ++iteration) {
      ZE_CHECK(launch());
      ze_kernel_timestamp_result_t timestamp{};
      ZE_CHECK(zeEventQueryKernelTimestamp(event, &timestamp));
      uint64_t cycles =
          (timestamp.global.kernelEnd - timestamp.global.kernelStart) &
          timestampMask;
      elapsedNanoseconds += cycles * properties.timerResolution;
      ZE_CHECK(zeEventHostReset(event));
    }
    samples.push_back(elapsedNanoseconds / iterations / 1000.0);
    if (!validateFull(batch + 1 == batches))
      return 1;
    if (batch + 1 == batches)
      continue;

    poisonOutput();
    ZE_CHECK(launch());
    if (!validateFull(false))
      return 1;
    ZE_CHECK(zeEventHostReset(event));
    ZE_CHECK(launch());
    ZE_CHECK(zeEventHostReset(event));
  }
  std::sort(samples.begin(), samples.end());
  size_t middle = samples.size() / 2;
  double median = samples.size() % 2
                      ? samples[middle]
                      : (samples[middle - 1] + samples[middle]) / 2.0;
  printf("%s median_us=%.6f min_us=%.6f max_us=%.6f tflops=%.6f\n", argv[1],
         median, samples.front(), samples.back(),
         (2.0 * m * n * k) / median / 1.0e6);

  ZE_CHECK(zeEventDestroy(event));
  ZE_CHECK(zeEventPoolDestroy(eventPool));
  ZE_CHECK(zeMemFree(context, aStorage));
  ZE_CHECK(zeMemFree(context, bStorage));
  ZE_CHECK(zeMemFree(context, cAllocation));
  ZE_CHECK(zeKernelDestroy(kernel));
  ZE_CHECK(zeModuleDestroy(module));
  ZE_CHECK(zeCommandListDestroy(commandList));
  ZE_CHECK(zeContextDestroy(context));
  return 0;
}
