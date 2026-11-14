// RUN: %clang_hip_dasan %s -o %t %hip_libs
// RUN: %env_dasan_opts=halt_on_error=0 %run %t 2>&1 \
// RUN:   | FileCheck %s --implicit-check-not=SUMMARY
// RUN: %clang_hip_dasan -O1 %s -o %t %hip_libs
// RUN: %env_dasan_opts=halt_on_error=0 %run %t 2>&1 \
// RUN:   | FileCheck %s --implicit-check-not=SUMMARY

__global__ void read_past(int *P, int *Out, long I) { *Out = P[I]; }

template <int> __global__ void write_at(int *P, long I) { P[I] = 1; }

__global__ void write_across(double *P, int I) { P[I] = 1.0; }

// The reported frame is the innermost function, not the kernel.
__attribute__((noinline)) __device__ void store(int *P, long I) { P[I] = 1; }
__global__ void call_store(int *P, long I) { store(P, I); }

// Bounds come from the allocation, not the interior pointer it was handed.
__global__ void walk(int *Q, int Hi) {
#pragma clang loop unroll(disable)
  for (int I = -32; I <= Hi; ++I)
    Q[I] = I;
}

__global__ void write_byte(char *P) { *P = 1; }

int main() {
  int *A = nullptr, *B = nullptr, *C = nullptr, *Out = nullptr;
  double *D = nullptr;
  CHECK_HIP(hipMalloc((void **)&A, 60 * sizeof(int)));
  CHECK_HIP(hipMalloc((void **)&B, 60 * sizeof(int)));
  CHECK_HIP(hipMalloc((void **)&C, 64 * sizeof(int)));
  CHECK_HIP(hipMalloc((void **)&D, 252));
  CHECK_HIP(hipMalloc((void **)&Out, sizeof(int)));
  int *Lo = A < B ? A : B;
  int *Hi = A < B ? B : A;

  read_past<<<1, 1>>>(C, Out, 64);
  write_at<0><<<1, 1>>>(C, -1);
  write_across<<<1, 1>>>(D, 31);
  write_at<1><<<1, 1>>>(A, -(1L << 20));
  write_at<2><<<1, 1>>>(Lo, Hi - Lo);
  call_store<<<1, 1>>>(C, 64);
  walk<<<1, 1>>>(C + 32, 31);
  walk<<<1, 1>>>(C + 32, 32);
  write_byte<<<1, 1>>>((char *)C + (1 << 20));
  CHECK_HIP(hipDeviceSynchronize());
  return 0;
}

// CHECK: ERROR: DeviceAddressSanitizer: heap-buffer-overflow on address [[ADDR:0x[0-9a-f]+]]
// CHECK: READ of size 4 at [[ADDR]]
// CHECK: #0 {{0x[0-9a-f]+}} in read_past{{.*}}heap-bounds.hip
// CHECK: [[ADDR]] is located 0 bytes after 256-byte region
// CHECK: SUMMARY: DeviceAddressSanitizer: heap-buffer-overflow

// CHECK: ERROR: DeviceAddressSanitizer: heap-buffer-underflow on address [[ADDR:0x[0-9a-f]+]]
// CHECK: WRITE of size 4 at [[ADDR]]
// CHECK: #0 {{0x[0-9a-f]+}} in {{.*}}write_at{{.*}}heap-bounds.hip
// CHECK: [[ADDR]] is located 4 bytes before 256-byte region
// CHECK: SUMMARY: DeviceAddressSanitizer: heap-buffer-underflow

// CHECK: ERROR: DeviceAddressSanitizer: heap-buffer-overflow on address [[ADDR:0x[0-9a-f]+]]
// CHECK: WRITE of size 8 at [[ADDR]]
// CHECK: #0 {{0x[0-9a-f]+}} in write_across{{.*}}heap-bounds.hip
// CHECK: [[ADDR]] is located 248 bytes inside of 252-byte region
// CHECK: SUMMARY: DeviceAddressSanitizer: heap-buffer-overflow

// CHECK: ERROR: DeviceAddressSanitizer: heap-buffer-underflow on address [[ADDR:0x[0-9a-f]+]]
// CHECK: #0 {{0x[0-9a-f]+}} in {{.*}}write_at{{.*}}heap-bounds.hip
// CHECK: [[ADDR]] is located 4194304 bytes before 240-byte region
// CHECK: SUMMARY: DeviceAddressSanitizer: heap-buffer-underflow

// CHECK: ERROR: DeviceAddressSanitizer: heap-buffer-overflow on address [[ADDR:0x[0-9a-f]+]]
// CHECK: #0 {{0x[0-9a-f]+}} in {{.*}}write_at{{.*}}heap-bounds.hip
// CHECK: [[ADDR]] is located {{[0-9]+}} bytes after 240-byte region
// CHECK: SUMMARY: DeviceAddressSanitizer: heap-buffer-overflow

// CHECK: ERROR: DeviceAddressSanitizer: heap-buffer-overflow on address [[ADDR:0x[0-9a-f]+]]
// CHECK: #0 {{0x[0-9a-f]+}} in store{{.*}}heap-bounds.hip
// CHECK: [[ADDR]] is located 0 bytes after 256-byte region
// CHECK: SUMMARY: DeviceAddressSanitizer: heap-buffer-overflow

// CHECK: ERROR: DeviceAddressSanitizer: heap-buffer-overflow on address [[ADDR:0x[0-9a-f]+]]
// CHECK: #0 {{0x[0-9a-f]+}} in walk{{.*}}heap-bounds.hip
// CHECK: [[ADDR]] is located 0 bytes after 256-byte region
// CHECK: SUMMARY: DeviceAddressSanitizer: heap-buffer-overflow

// CHECK: ERROR: DeviceAddressSanitizer: unknown-device-address on address [[ADDR:0x[0-9a-f]+]]
// CHECK: WRITE of size 1 at [[ADDR]]
// CHECK: #0 {{0x[0-9a-f]+}} in write_byte{{.*}}heap-bounds.hip
// CHECK: SUMMARY: DeviceAddressSanitizer: unknown-device-address
