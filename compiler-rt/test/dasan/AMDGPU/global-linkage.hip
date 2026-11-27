// REQUIRES: dasan-globals
// RUN: %clang_hip_dasan %s -o %t %hip_libs
// RUN: %env_dasan_opts=halt_on_error=0 %run %t 2>&1 \
// RUN:   | FileCheck %s --implicit-check-not=SUMMARY
// RUN: %clang_hip_dasan -O3 %s -o %t %hip_libs
// RUN: %env_dasan_opts=halt_on_error=0 %run %t 2>&1 \
// RUN:   | FileCheck %s --implicit-check-not=SUMMARY

// COMDAT and internal globals are placed like any other.
inline __device__ int Inline[16];
inline __device__ char Tail[100];
static __device__ int Local[16];

__device__ __attribute__((noinline)) void fill(int *Begin, int *End) {
  for (int *P = End; P != Begin;)
    *--P = 1;
}

__global__ void fill_inline() { fill(Inline, Inline + 16); }

__global__ void write_tail(int I) { Tail[I] = 1; }

__global__ void write_local(int *Out, int I) {
  Local[I] = 1;
  *Out = Local[2];
}

int main() {
  int *Out = nullptr;
  CHECK_HIP(hipMalloc((void **)&Out, sizeof(int)));

  fill_inline<<<1, 1>>>();
  write_tail<<<1, 1>>>(99);
  write_tail<<<1, 1>>>(100);
  write_local<<<1, 1>>>(Out, 16);
  CHECK_HIP(hipDeviceSynchronize());
  return 0;
}

// CHECK: ERROR: DeviceAddressSanitizer: global-buffer-overflow on address [[ADDR:0x[0-9a-f]+]]
// CHECK: WRITE of size 1 at [[ADDR]]
// CHECK: #0 {{0x[0-9a-f]+}} in write_tail{{.*}}global-linkage.hip
// CHECK: [[ADDR]] is located 0 bytes after global variable 'Tail' {{.*}} of size 100
// CHECK: SUMMARY: DeviceAddressSanitizer: global-buffer-overflow

// CHECK: ERROR: DeviceAddressSanitizer: global-buffer-overflow on address [[ADDR:0x[0-9a-f]+]]
// CHECK: WRITE of size 4 at [[ADDR]]
// CHECK: #0 {{0x[0-9a-f]+}} in write_local{{.*}}global-linkage.hip
// CHECK: [[ADDR]] is located 0 bytes after global variable '{{.*}}Local' {{.*}} of size 64
// CHECK: SUMMARY: DeviceAddressSanitizer: global-buffer-overflow
