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
  if (argc != 9 ||
      (std::string(argv[1]) != "inter" && std::string(argv[1]) != "igc")) {
    fprintf(stderr,
            "usage: %s inter|igc <zebin> <device> <warmups> <batches> "
            "<iterations> <size> <kernel>\n",
            argv[0]);
    return 1;
  }
  const bool inter = std::string(argv[1]) == "inter";
  const int warmups = std::atoi(argv[4]);
  const int batches = std::atoi(argv[5]);
  const int iterations = std::atoi(argv[6]);
  const int64_t size = std::atoll(argv[7]);
  if (warmups < 1 || batches < 1 || iterations < 1) {
    fprintf(stderr, "warmups, batches, and iterations must be positive\n");
    return 1;
  }
  if (size < 64 || size % 64 != 0) {
    fprintf(stderr, "matrix size must be a positive multiple of 64\n");
    return 1;
  }
  const int64_t m = size;
  const int64_t n = size;
  constexpr int64_t k = 64;

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
  kernelDesc.pKernelName = argv[8];
  ze_kernel_handle_t kernel;
  ZE_CHECK(zeKernelCreate(module, &kernelDesc, &kernel));
  ZE_CHECK(zeKernelSetGroupSize(kernel, 256, 1, 1));

  ze_device_mem_alloc_desc_t deviceAlloc{
      ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC};
  ze_host_mem_alloc_desc_t hostAlloc{ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC};
  void *aStorage = nullptr;
  void *bStorage = nullptr;
  void *cStorage = nullptr;
  ZE_CHECK(zeMemAllocShared(context, &deviceAlloc, &hostAlloc,
                            m * k * sizeof(_Float16), 64, device, &aStorage));
  ZE_CHECK(zeMemAllocShared(context, &deviceAlloc, &hostAlloc,
                            k * n * sizeof(_Float16), 64, device, &bStorage));
  ZE_CHECK(zeMemAllocShared(context, &deviceAlloc, &hostAlloc,
                            m * n * sizeof(float), 64, device, &cStorage));
  auto *a = static_cast<_Float16 *>(aStorage);
  auto *b = static_cast<_Float16 *>(bStorage);
  auto *c = static_cast<float *>(cStorage);
  std::mt19937 random(0x4D41544Du);
  std::uniform_int_distribution<int> distribution(-8, 8);
  for (int64_t index = 0; index < m * k; ++index)
    a[index] = static_cast<_Float16>(distribution(random) * 0.125f);
  for (int64_t index = 0; index < k * n; ++index)
    b[index] = static_cast<_Float16>(distribution(random) * 0.125f);
  std::fill(c, c + m * n, std::numeric_limits<float>::quiet_NaN());

  if (inter) {
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
  } else {
    ZE_CHECK(zeKernelSetArgumentValue(kernel, 0, sizeof(void *), &aStorage));
    ZE_CHECK(zeKernelSetArgumentValue(kernel, 1, sizeof(void *), &bStorage));
    ZE_CHECK(zeKernelSetArgumentValue(kernel, 2, sizeof(void *), &cStorage));
  }

  std::vector<float> reference(m * n, 0.0f);
  for (int64_t row = 0; row < m; ++row)
    for (int64_t column = 0; column < n; ++column)
      for (int64_t inner = 0; inner < k; ++inner)
        reference[row * n + column] +=
            static_cast<float>(a[row * k + inner]) *
            static_cast<float>(b[inner * n + column]);

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
  auto validate = [&]() {
    for (int64_t index = 0; index < m * n; ++index)
      if (!std::isfinite(c[index]) || c[index] != reference[index]) {
        fprintf(stderr, "incorrect C[%ld,%ld]: %.9g != %.9g\n", index / n,
                index % n, c[index], reference[index]);
        return false;
      }
    return true;
  };

  for (int iteration = 0; iteration < warmups; ++iteration) {
    ZE_CHECK(launch());
    if (!validate())
      return 1;
    ZE_CHECK(zeEventHostReset(event));
  }

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
  }
  if (!validate())
    return 1;
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
  ZE_CHECK(zeMemFree(context, cStorage));
  ZE_CHECK(zeKernelDestroy(kernel));
  ZE_CHECK(zeModuleDestroy(module));
  ZE_CHECK(zeCommandListDestroy(commandList));
  ZE_CHECK(zeContextDestroy(context));
  return 0;
}
