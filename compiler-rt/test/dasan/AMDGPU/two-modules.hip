// RUN: %clang_hip_dasan %s %S/../Inputs/second-module.hip -o %t %hip_libs
// RUN: %env_dasan_opts=halt_on_error=0 %run %t 2>&1 | FileCheck %s \
// RUN:   --implicit-check-not=SUMMARY \
// RUN:   %if dasan-globals %{ --check-prefixes=GLOBAL,SHARED %} %else %{ --check-prefix=SHARED %}
// RUN: %clang_hip_dasan -O1 %s %S/../Inputs/second-module.hip -o %t %hip_libs
// RUN: %env_dasan_opts=halt_on_error=0 %run %t 2>&1 | FileCheck %s \
// RUN:   --implicit-check-not=SUMMARY \
// RUN:   %if dasan-globals %{ --check-prefixes=GLOBAL,SHARED %} %else %{ --check-prefix=SHARED %}

// Everything the pass adds is per module, so two instrumented translation
// units in one image must coexist and keep their own extents.
extern __global__ void write_other(int I);
extern __global__ void write_other_shared(int *Out, int I);

__device__ int Mine[16];

__global__ void write_mine(int I) { Mine[I] = 1; }

int main() {
  int *Out = nullptr;
  CHECK_HIP(hipMalloc((void **)&Out, sizeof(int)));

  write_mine<<<1, 1>>>(15);
  write_other<<<1, 1>>>(15);
  write_other_shared<<<1, 16>>>(Out, 15);
  write_mine<<<1, 1>>>(16);
  write_other<<<1, 1>>>(16);
  write_other_shared<<<1, 16>>>(Out, 16);
  CHECK_HIP(hipDeviceSynchronize());
  return 0;
}

// GLOBAL: ERROR: DeviceAddressSanitizer: global-buffer-overflow
// GLOBAL: in write_mine{{.*}}two-modules.hip
// GLOBAL: is located 0 bytes after global variable 'Mine' {{.*}} of size 64
// GLOBAL: SUMMARY: DeviceAddressSanitizer: global-buffer-overflow

// GLOBAL: ERROR: DeviceAddressSanitizer: global-buffer-overflow
// GLOBAL: in write_other{{.*}}second-module.hip
// GLOBAL: is located 0 bytes after global variable 'Other' {{.*}} of size 64
// GLOBAL: SUMMARY: DeviceAddressSanitizer: global-buffer-overflow

// SHARED: ERROR: DeviceAddressSanitizer: shared-buffer-overflow
// SHARED: in write_other_shared{{.*}}second-module.hip
// SHARED: is located 0 bytes after 64-byte region
// SHARED: SUMMARY: DeviceAddressSanitizer: shared-buffer-overflow
