// Container proof launcher: wrap a zebin in an OffloadBinary, load via
// liboffload, launch on the GPU, verify out[i] == i*mul + bias.
//
// Usage: launcher <zebin.bin> <kernel-name> <num-work-items> [bias] [mul]
//
// No Level Zero calls here by design; liboffload owns the host path.

#include <OffloadAPI.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CHECK(expr)                                                          \
  do {                                                                       \
    ol_result_t res_ = (expr);                                               \
    if (res_ != OL_SUCCESS) {                                                \
      fprintf(stderr, "FAIL %s:%d: %s -> %d\n", __FILE__, __LINE__, #expr,   \
              res_->Code);                                              \
      return 1;                                                              \
    }                                                                        \
  } while (0)

// Wrap image bytes in an OffloadBinary (v2) container: one entry,
// IMG_Object + spirv64 triple, matching what the L0 plugin validates.
static std::vector<uint8_t> wrapOffloadBinary(const std::vector<uint8_t> &image) {
  static const char key[] = "triple";
  static const char triple[] = "spirv64-unknown-unknown";

  const uint64_t headerSize = 40;      // magic + version + size + off + count
  const uint64_t entrySize = 48;       // Entry struct
  const uint64_t stringEntSize = 24;   // one StringEntry
  const uint64_t stringsOff = headerSize + entrySize + stringEntSize; // 112
  const uint64_t keyOff = stringsOff;                        // 112
  const uint64_t tripleOff = keyOff + sizeof(key);           // 119
  const uint64_t imageOff = (tripleOff + sizeof(triple) - 1 + 7) & ~7ull; // 144

  std::vector<uint8_t> blob(imageOff + image.size(), 0);
  uint8_t *p = blob.data();

  // Header.
  p[0] = 0x10; p[1] = 0xFF; p[2] = 0x10; p[3] = 0xAD;
  auto put32 = [&](uint64_t off, uint32_t v) { memcpy(p + off, &v, 4); };
  auto put64 = [&](uint64_t off, uint64_t v) { memcpy(p + off, &v, 8); };
  put32(4, 2);                       // version
  put64(8, blob.size());             // total size
  put64(16, headerSize);             // entries offset
  put64(24, 1);                      // entries count

  // Entry: IMG_Object=1, OFK_SYCL=8.
  put32(40, (uint32_t(8) << 16) | 1u);
  put32(44, 0);                      // flags
  put64(48, headerSize + entrySize); // string map offset (88)
  put64(56, 1);                      // num strings
  put64(64, imageOff);
  put64(72, image.size());

  // StringEntry.
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
  if (!f) { perror(path); exit(1); }
  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  fseek(f, 0, SEEK_SET);
  std::vector<uint8_t> buf(sz);
  if (fread(buf.data(), 1, sz, f) != (size_t)sz) { perror("read"); exit(1); }
  fclose(f);
  return buf;
}

static ol_device_handle_t pickL0GpuDevice() {
  struct Ctx { ol_device_handle_t found = nullptr; };
  Ctx ctx;
  olIterateDevices(
      [](ol_device_handle_t dev, void *data) -> bool {
        auto *c = static_cast<Ctx *>(data);
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
        fprintf(stderr, "picking L0 GPU device: %s\n", name);
        c->found = dev;
        return false;
      },
      &ctx);
  return ctx.found;
}

int main(int argc, char **argv) {
  if (argc < 4) {
    fprintf(stderr, "usage: %s <zebin.bin> <kernel-name> <n> [bias]\n",
            argv[0]);
    return 1;
  }
  const char *zebinPath = argv[1];
  const char *kernelName = argv[2];
  size_t n = strtoul(argv[3], nullptr, 0);
  uint32_t bias = argc > 4 ? strtoul(argv[4], nullptr, 0) : 7;
  uint32_t mul = argc > 5 ? strtoul(argv[5], nullptr, 0) : 2;

  CHECK(olInit(nullptr));

  ol_device_handle_t dev = pickL0GpuDevice();
  if (!dev) {
    fprintf(stderr, "FAIL: no Level Zero GPU device found\n");
    return 1;
  }

  std::vector<uint8_t> zebin = readFile(zebinPath);
  std::vector<uint8_t> blob = wrapOffloadBinary(zebin);

  bool valid = false;
  CHECK(olIsValidBinary(dev, blob.data(), blob.size(), &valid));
  fprintf(stderr, "olIsValidBinary: %s\n", valid ? "true" : "false");
  if (!valid)
    return 1;

  ol_program_handle_t prog;
  CHECK(olCreateProgram(dev, blob.data(), blob.size(), &prog));

  ol_symbol_handle_t kern;
  CHECK(olGetSymbol(prog, kernelName, OL_SYMBOL_KIND_KERNEL, &kern));

  void *outBuf = nullptr;
  CHECK(olMemAlloc(dev, OL_ALLOC_TYPE_MANAGED, n * sizeof(uint32_t), &outBuf));
  memset(outBuf, 0xA5, n * sizeof(uint32_t));

  void *argPtr = outBuf;
  void *argPtrs[2] = {&argPtr, &bias};
  size_t argSizes[2] = {sizeof(void *), sizeof(uint32_t)};

  ol_kernel_launch_size_args_t lsa;
  lsa.Dimensions = 1;
  lsa.NumGroups = {(n + 31) / 32, 1, 1};
  lsa.GroupSize = {32, 1, 1};
  lsa.DynSharedMemory = 0;

  CHECK(olLaunchKernel(nullptr, dev, kern, &lsa, nullptr, 2, argPtrs,
                       argSizes));

  size_t bad = 0;
  auto *out = static_cast<uint32_t *>(outBuf);
  for (size_t i = 0; i < n; ++i) {
    uint32_t want = uint32_t(i) * mul + bias;
    if (out[i] != want) {
      if (bad < 8)
        fprintf(stderr, "  out[%zu] = 0x%08x, want 0x%08x\n", i, out[i],
                want);
      ++bad;
    }
  }
  if (bad) {
    fprintf(stderr, "FAIL: %zu/%zu mismatches\n", bad, n);
    return 1;
  }
  printf("PASS: %zu lanes, out[i] == i*%u + %u on %s\n", n, mul, bias,
         kernelName);

  olMemFree(outBuf);
  olDestroyProgram(prog);
  olShutDown();
  return 0;
}
