// Generic zebin runner: wraps a zebin in an OffloadBinary, loads via
// liboffload, launches, dumps output buffers. No kernel knowledge.
//
// Usage: inter-runner [--compact] [--sort-output] [--group-size <n>]
//                     <zebin.bin> <kernel-name> <n> <arg-spec>...
//        inter-runner --probe [device-name-substring]
//   arg-spec (in kernel arg order):
//     out        device buffer, printed as hex after the run
//     in:<mul>   device buffer filled with i*mul before the run
//     inout:<mul> input buffer also printed after the run.
//     u32:<v>    scalar 32-bit argument with value v
//
// Output lines: "out<k>[<i>] = 0x........" for each out buffer.
// Default output can be consumed by verify.py; --compact is intended for
// FileCheck tests.

#include <OffloadAPI.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CHECK(expr)                                                            \
  do {                                                                         \
    ol_result_t res_ = (expr);                                                 \
    if (res_ != OL_SUCCESS) {                                                  \
      fprintf(stderr, "FAIL %s:%d: %s -> %d\n", __FILE__, __LINE__, #expr,     \
              res_->Code);                                                     \
      return 1;                                                                \
    }                                                                          \
  } while (0)

static std::vector<uint8_t>
wrapOffloadBinary(const std::vector<uint8_t> &image) {
  static const char key[] = "triple";
  static const char triple[] = "spirv64-unknown-unknown";

  const uint64_t headerSize = 40;
  const uint64_t entrySize = 48;
  const uint64_t stringEntSize = 24;
  const uint64_t keyOff = headerSize + entrySize + stringEntSize;
  const uint64_t tripleOff = keyOff + sizeof(key);
  const uint64_t imageOff = (tripleOff + sizeof(triple) - 1 + 7) & ~7ull;

  std::vector<uint8_t> blob(imageOff + image.size(), 0);
  uint8_t *p = blob.data();
  p[0] = 0x10;
  p[1] = 0xFF;
  p[2] = 0x10;
  p[3] = 0xAD;
  auto put32 = [&](uint64_t off, uint32_t v) { memcpy(p + off, &v, 4); };
  auto put64 = [&](uint64_t off, uint64_t v) { memcpy(p + off, &v, 8); };
  put32(4, 2);
  put64(8, blob.size());
  put64(16, headerSize);
  put64(24, 1);
  put32(40, (uint32_t(8) << 16) | 1u); // IMG_Object=1, OFK_SYCL=8
  put32(44, 0);
  put64(48, headerSize + entrySize);
  put64(56, 1);
  put64(64, imageOff);
  put64(72, image.size());
  put64(88, keyOff);
  put64(96, tripleOff);
  put64(104, sizeof(triple) - 1);
  memcpy(p + keyOff, key, sizeof(key));
  memcpy(p + tripleOff, triple, sizeof(triple) - 1);
  memcpy(p + imageOff, image.data(), image.size());
  return blob;
}

static std::vector<uint8_t> readFile(const char *path) {
  FILE *f = fopen(path, "rb");
  if (!f) {
    perror(path);
    exit(1);
  }
  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  fseek(f, 0, SEEK_SET);
  std::vector<uint8_t> buf(sz);
  if (fread(buf.data(), 1, sz, f) != (size_t)sz) {
    perror("read");
    exit(1);
  }
  fclose(f);
  return buf;
}

static ol_device_handle_t pickL0GpuDevice(const char *requiredName) {
  struct Ctx {
    ol_device_handle_t found = nullptr;
  };
  Ctx ctx;
  struct Search {
    Ctx *ctx;
    const char *requiredName;
  } search{&ctx, requiredName};
  olIterateDevices(
      [](ol_device_handle_t dev, void *data) -> bool {
        auto *search = static_cast<Search *>(data);
        ol_device_type_t type;
        if (olGetDeviceInfo(dev, OL_DEVICE_INFO_TYPE, sizeof(type), &type) !=
                OL_SUCCESS ||
            type != OL_DEVICE_TYPE_GPU)
          return true;
        ol_platform_handle_t plat;
        if (olGetDeviceInfo(dev, OL_DEVICE_INFO_PLATFORM, sizeof(plat),
                            &plat) != OL_SUCCESS)
          return true;
        ol_platform_backend_t backend;
        if (olGetPlatformInfo(plat, OL_PLATFORM_INFO_BACKEND, sizeof(backend),
                              &backend) != OL_SUCCESS)
          return true;
        if (backend != OL_PLATFORM_BACKEND_LEVEL_ZERO)
          return true;
        char name[256] = {0};
        olGetDeviceInfo(dev, OL_DEVICE_INFO_NAME, sizeof(name), name);
        if (search->requiredName && search->requiredName[0] != '\0' &&
            !strstr(name, search->requiredName))
          return true;
        fprintf(stderr, "device: %s\n", name);
        search->ctx->found = dev;
        return false;
      },
      &search);
  return ctx.found;
}

int main(int argc, char **argv) {
  if (argc >= 2 && std::string(argv[1]) == "--probe") {
    if (argc > 3) {
      fprintf(stderr, "usage: %s --probe [device-name-substring]\n", argv[0]);
      return 1;
    }
    const char *requiredName =
        argc == 3 ? argv[2] : getenv("INTER_DEVICE_NAME");
    CHECK(olInit(nullptr));
    ol_device_handle_t dev = pickL0GpuDevice(requiredName);
    if (!dev) {
      fprintf(stderr, "FAIL: no matching Level Zero GPU device found\n");
      olShutDown();
      return 1;
    }
    CHECK(olShutDown());
    return 0;
  }

  bool compactOutput = false;
  bool sortOutput = false;
  uint32_t groupSize = 32;
  int firstArg = 1;
  while (firstArg < argc) {
    std::string option = argv[firstArg];
    if (option == "--compact")
      compactOutput = true;
    else if (option == "--sort-output")
      sortOutput = true;
    else if (option == "--group-size") {
      if (++firstArg == argc) {
        fprintf(stderr, "FAIL: --group-size requires a value\n");
        return 1;
      }
      groupSize = strtoul(argv[firstArg], nullptr, 0);
      if (groupSize == 0) {
        fprintf(stderr, "FAIL: group size must be nonzero\n");
        return 1;
      }
    } else
      break;
    ++firstArg;
  }

  if (argc - firstArg < 3) {
    fprintf(stderr,
            "usage: %s [--compact] [--sort-output] [--group-size <n>] "
            "<zebin.bin> <kernel> <n> <spec>...\n"
            "  spec: out | in:<mul> | inout:<mul> | u32:<value>\n",
            argv[0]);
    return 1;
  }
  const char *zebinPath = argv[firstArg];
  const char *kernelName = argv[firstArg + 1];
  size_t n = strtoul(argv[firstArg + 2], nullptr, 0);
  if (n == 0 || n % groupSize != 0) {
    fprintf(stderr,
            "FAIL: launch size must be a nonzero multiple of group size\n");
    return 1;
  }
  int numArgs = argc - firstArg - 3;
  char **specs = argv + firstArg + 3;

  CHECK(olInit(nullptr));
  ol_device_handle_t dev = pickL0GpuDevice(getenv("INTER_DEVICE_NAME"));
  if (!dev) {
    fprintf(stderr, "FAIL: no Level Zero GPU device found\n");
    return 1;
  }

  std::vector<uint8_t> blob = wrapOffloadBinary(readFile(zebinPath));
  bool valid = false;
  CHECK(olIsValidBinary(dev, blob.data(), blob.size(), &valid));
  fprintf(stderr, "olIsValidBinary: %s\n", valid ? "true" : "false");
  if (!valid)
    return 1;

  ol_program_handle_t prog;
  CHECK(olCreateProgram(dev, blob.data(), blob.size(), &prog));
  ol_symbol_handle_t kern;
  CHECK(olGetSymbol(prog, kernelName, OL_SYMBOL_KIND_KERNEL, &kern));

  std::vector<void *> argPtrsStorage(numArgs);
  std::vector<void *> argPtrs(numArgs);
  std::vector<size_t> argSizes(numArgs);
  std::vector<uint32_t> scalars(numArgs);
  std::vector<int> outIndex(numArgs, -1);
  std::vector<bool> bufferArgument(numArgs, false);
  int numOuts = 0;

  for (int i = 0; i < numArgs; ++i) {
    std::string spec = specs[i];
    bool isInput = spec.rfind("in:", 0) == 0;
    bool isInputOutput = spec.rfind("inout:", 0) == 0;
    if (spec == "out" || isInput || isInputOutput) {
      void *buf = nullptr;
      CHECK(olMemAlloc(dev, OL_ALLOC_TYPE_MANAGED, n * sizeof(uint32_t), &buf));
      auto *w = static_cast<uint32_t *>(buf);
      if (spec == "out") {
        for (size_t j = 0; j < n; ++j)
          w[j] = 0xA5A5A5A5u;
      } else {
        size_t prefixLength = isInputOutput ? 6 : 3;
        uint32_t mul = strtoul(spec.c_str() + prefixLength, nullptr, 0);
        for (size_t j = 0; j < n; ++j)
          w[j] = uint32_t(j) * mul;
      }
      if (spec == "out" || isInputOutput)
        outIndex[i] = numOuts++;
      argPtrsStorage[i] = buf;
      argPtrs[i] = &argPtrsStorage[i];
      argSizes[i] = sizeof(void *);
      bufferArgument[i] = true;
    } else if (spec.rfind("u32:", 0) == 0) {
      scalars[i] = strtoul(spec.c_str() + 4, nullptr, 0);
      argPtrs[i] = &scalars[i];
      argSizes[i] = sizeof(uint32_t);
    } else {
      fprintf(stderr, "FAIL: bad spec '%s'\n", spec.c_str());
      return 1;
    }
  }

  ol_kernel_launch_size_args_t lsa;
  lsa.Dimensions = 1;
  lsa.NumGroups = {uint32_t(n / groupSize), 1, 1};
  lsa.GroupSize = {groupSize, 1, 1};
  lsa.DynSharedMemory = 0;
  CHECK(olLaunchKernel(nullptr, dev, kern, &lsa, nullptr, numArgs,
                       argPtrs.data(), argSizes.data()));

  for (int i = 0; i < numArgs; ++i) {
    if (outIndex[i] < 0)
      continue;
    auto *w = static_cast<uint32_t *>(argPtrsStorage[i]);
    if (sortOutput)
      std::sort(w, w + n);
    if (compactOutput) {
      printf("out%d = [", outIndex[i]);
      for (size_t j = 0; j < n; ++j)
        printf("%s0x%08x", j == 0 ? "" : ", ", w[j]);
      printf("]\n");
    } else {
      for (size_t j = 0; j < n; ++j)
        printf("out%d[%zu] = 0x%08x\n", outIndex[i], j, w[j]);
    }
  }

  for (int i = 0; i < numArgs; ++i)
    if (bufferArgument[i])
      CHECK(olMemFree(argPtrsStorage[i]));
  CHECK(olDestroyProgram(prog));
  CHECK(olShutDown());
  return 0;
}
