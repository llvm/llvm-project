// RUN: %clang_hip_dasan %s -o %t %hip_libs
// RUN: %env_dasan_opts=halt_on_error=0 %run %t 2>&1 \
// RUN:   | FileCheck %s --implicit-check-not=SUMMARY
// RUN: %clang_hip_dasan -O1 %s -o %t %hip_libs
// RUN: %env_dasan_opts=halt_on_error=0 %run %t 2>&1 \
// RUN:   | FileCheck %s --implicit-check-not=SUMMARY

// Group objects are measured against their own extent, not the segment's.
__global__ void write_past(int *Out, int I) {
  __shared__ int Buf[16];
  __shared__ int Other[16];
  unsigned T = threadIdxX();
  Buf[T] = T;
  Other[T] = 0;
  syncThreads();
  if (T == 0)
    Buf[I] = 0x41;
  syncThreads();
  *Out = Buf[0] + Other[0];
}

__global__ void write_before(int *Out, int I) {
  __shared__ int First[16];
  __shared__ int Second[16];
  unsigned T = threadIdxX();
  First[T] = T;
  Second[T] = T;
  syncThreads();
  if (T == 0)
    Second[I] = 0x41;
  syncThreads();
  *Out = First[0] + Second[0];
}

// The first object sits at offset 0, so a negative index wraps the 32-bit LDS
// address. Needs to be reported as underflow.
__global__ void write_under_zero(int *Out, int I) {
  __shared__ int Buf[4];
  Buf[0] = 1;
  Buf[I] = 9;
  *Out = Buf[0];
}

// The extent of dynamic group memory comes from the dispatch packet.
__global__ void write_dynamic(int *Out, int I) {
  extern __shared__ int Dyn[];
  unsigned T = threadIdxX();
  Dyn[T] = T;
  syncThreads();
  if (T == 0)
    Dyn[I] = 0x41;
  syncThreads();
  *Out = Dyn[0];
}

// TODO: An LDS pointer that escapes into a callee is checked as heap, which
// does not cover the group segment, so these are not reported.
__device__ __attribute__((noinline)) void poke(int *P, int I) { P[I] = 0x41; }
__device__ __attribute__((noinline)) void poke0(int *P) { *P = 0x41; }

__global__ void escape(int *Out, int I) {
  __shared__ int Buf[16];
  __shared__ int Other[16];
  unsigned T = threadIdxX();
  Buf[T] = T;
  Other[T] = 0;
  syncThreads();
  if (T == 0) {
    poke(Buf, I);
    poke0(Buf + I);
  }
  syncThreads();
  *Out = Buf[0] + Other[0];
}

int main() {
  int *Out = nullptr;
  CHECK_HIP(hipMalloc((void **)&Out, sizeof(int)));

  write_past<<<1, 16>>>(Out, 16);
  write_before<<<1, 16>>>(Out, -1);
  write_under_zero<<<1, 1>>>(Out, -1);
  write_dynamic<<<1, 8, 8 * sizeof(int)>>>(Out, 8);
  escape<<<1, 16>>>(Out, 16);
  CHECK_HIP(hipDeviceSynchronize());
  return 0;
}

// CHECK: ERROR: DeviceAddressSanitizer: shared-buffer-overflow on address [[ADDR:0x[0-9a-f]+]]
// CHECK: WRITE of size 4 at [[ADDR]]
// CHECK: #0 {{0x[0-9a-f]+}} in write_past{{.*}}shared-bounds.hip
// CHECK: [[ADDR]] is located 0 bytes after 64-byte region
// CHECK: SUMMARY: DeviceAddressSanitizer: shared-buffer-overflow

// CHECK: ERROR: DeviceAddressSanitizer: shared-buffer-underflow on address [[ADDR:0x[0-9a-f]+]]
// CHECK: WRITE of size 4 at [[ADDR]]
// CHECK: #0 {{0x[0-9a-f]+}} in write_before{{.*}}shared-bounds.hip
// CHECK: [[ADDR]] is located 4 bytes before 64-byte region
// CHECK: the region is group memory
// CHECK: SUMMARY: DeviceAddressSanitizer: shared-buffer-underflow

// CHECK: ERROR: DeviceAddressSanitizer: shared-buffer-underflow on address [[ADDR:0x[0-9a-f]+]]
// CHECK: WRITE of size 4 at [[ADDR]]
// CHECK: #0 {{0x[0-9a-f]+}} in write_under_zero{{.*}}shared-bounds.hip
// CHECK: [[ADDR]] is located 4 bytes before 16-byte region
// CHECK: SUMMARY: DeviceAddressSanitizer: shared-buffer-underflow

// CHECK: ERROR: DeviceAddressSanitizer: shared-buffer-overflow on address [[ADDR:0x[0-9a-f]+]]
// CHECK: WRITE of size 4 at [[ADDR]]
// CHECK: #0 {{0x[0-9a-f]+}} in write_dynamic{{.*}}shared-bounds.hip
// CHECK: [[ADDR]] is located 0 bytes after 32-byte region
// CHECK: SUMMARY: DeviceAddressSanitizer: shared-buffer-overflow
