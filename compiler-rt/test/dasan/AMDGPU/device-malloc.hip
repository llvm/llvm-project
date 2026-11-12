// RUN: %clang_hip_dasan %s -o %t %hip_libs
// RUN: not %run %t 2>&1 | FileCheck %s --check-prefix=MALLOC
// RUN: not %run %t free 2>&1 | FileCheck %s --check-prefix=FREE
// RUN: %clang_hip_dasan -O1 %s -o %t %hip_libs
// RUN: not %run %t 2>&1 | FileCheck %s --check-prefix=MALLOC
// RUN: not %run %t free 2>&1 | FileCheck %s --check-prefix=FREE
// RUN: %clang_hip_dasan -shared-libsan %s -o %t %hip_libs
// RUN: not %run %t 2>&1 | FileCheck %s --check-prefix=MALLOC
// RUN: not %run %t free 2>&1 | FileCheck %s --check-prefix=FREE

__global__ void device_malloc() {
  void *P = malloc(16);
  (void)P;
}

__global__ void device_free() { free(nullptr); }

int main(int argc, char **argv) {
  if (argc > 1 && argv[1][0] == 'f')
    device_free<<<1, 1>>>();
  else
    device_malloc<<<1, 1>>>();
  CHECK_HIP(hipDeviceSynchronize());
  return 0;
}

// MALLOC: ERROR: DeviceAddressSanitizer: device malloc is not supported
// MALLOC: #0 {{0x[0-9a-f]+}} in device_malloc{{.*}}device-malloc.hip
// MALLOC: SUMMARY: DeviceAddressSanitizer: unsupported-device-malloc

// FREE: ERROR: DeviceAddressSanitizer: device free is not supported
// FREE: #0 {{0x[0-9a-f]+}} in device_free{{.*}}device-malloc.hip
// FREE: SUMMARY: DeviceAddressSanitizer: unsupported-device-free
