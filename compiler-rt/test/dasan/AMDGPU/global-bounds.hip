// REQUIRES: dasan-globals
// RUN: %clang_hip_dasan %s -o %t %hip_libs
// RUN: %env_dasan_opts=halt_on_error=0 %run %t 2>&1 \
// RUN:   | FileCheck %s --implicit-check-not=SUMMARY
// RUN: %clang_hip_dasan -O1 %s -o %t %hip_libs
// RUN: %env_dasan_opts=halt_on_error=0 %run %t 2>&1 \
// RUN:   | FileCheck %s --implicit-check-not=SUMMARY

__device__ int Before[256];
__device__ int Table[256];
__device__ int Pointee[16];
__device__ int *Ptr;
__constant__ int Konst[16] = {1};

__device__ __attribute__((aligned(1024))) int Aligned[4];

__device__ int Wide[1024];
__device__ int Small[32];
__device__ int Big[4096];

constexpr long N = 4L << 20;
__device__ int Huge[N];

__global__ void read_before(int I) { Before[0] = Table[I]; }

__global__ void publish() { Ptr = Pointee; }
__global__ void write_indirect(int I) { Ptr[I] = 1; }

__global__ void read_const(int *Out, int I) { *Out = Konst[I]; }

__global__ void write_aligned(int I) { Aligned[I] = 1; }
__global__ void write_wide(int I) { Wide[I] = 1; }
__global__ void write_small(int I) { Small[I] = 1; }
__global__ void write_big(int I) { Big[I] = 1; }

__global__ void write_huge(int *Q, long Lo, long Hi) {
  Q[Lo] = 1;
  Q[Hi] = 2;
}

int main() {
  int *Out = nullptr, *H = nullptr;
  CHECK_HIP(hipMalloc((void **)&Out, sizeof(int)));
  CHECK_HIP(hipGetSymbolAddress((void **)&H, (const void *)&Huge));

  read_before<<<1, 1>>>(-1);
  publish<<<1, 1>>>();
  write_indirect<<<1, 1>>>(16);
  read_const<<<1, 1>>>(Out, 16);
  write_aligned<<<1, 1>>>(3);
  write_aligned<<<1, 1>>>(4);
  write_wide<<<1, 1>>>(1023);
  write_wide<<<1, 1>>>(1024);
  write_small<<<1, 1>>>(31);
  write_small<<<1, 1>>>(40);
  write_big<<<1, 1>>>(4095);
  write_big<<<1, 1>>>(4608);
  write_huge<<<1, 1>>>(H + N - 16, -(N - 16), 15);
  write_huge<<<1, 1>>>(H + N - 16, -(N - 16), 16);
  CHECK_HIP(hipDeviceSynchronize());
  return 0;
}

// CHECK: ERROR: DeviceAddressSanitizer: global-buffer-underflow on address [[ADDR:0x[0-9a-f]+]]
// CHECK: READ of size 4 at [[ADDR]]
// CHECK: #0 {{0x[0-9a-f]+}} in read_before{{.*}}global-bounds.hip
// CHECK: [[ADDR]] is located 4 bytes before global variable 'Table' {{.*}} of size 1024
// CHECK: SUMMARY: DeviceAddressSanitizer: global-buffer-underflow

// CHECK: ERROR: DeviceAddressSanitizer: global-buffer-overflow on address [[ADDR:0x[0-9a-f]+]]
// CHECK: #0 {{0x[0-9a-f]+}} in write_indirect{{.*}}global-bounds.hip
// CHECK: [[ADDR]] is located 0 bytes after global variable 'Pointee' {{.*}} of size 64
// CHECK: SUMMARY: DeviceAddressSanitizer: global-buffer-overflow

// CHECK: ERROR: DeviceAddressSanitizer: global-buffer-overflow on address [[ADDR:0x[0-9a-f]+]]
// CHECK: READ of size 4 at [[ADDR]]
// CHECK: #0 {{0x[0-9a-f]+}} in read_const{{.*}}global-bounds.hip
// CHECK: [[ADDR]] is located 0 bytes after global variable 'Konst' {{.*}} of size 64
// CHECK: SUMMARY: DeviceAddressSanitizer: global-buffer-overflow

// CHECK: ERROR: DeviceAddressSanitizer: global-buffer-overflow on address [[ADDR:0x[0-9a-f]+]]
// CHECK: #0 {{0x[0-9a-f]+}} in write_aligned{{.*}}global-bounds.hip
// CHECK: [[ADDR]] is located 0 bytes after global variable 'Aligned' {{.*}} of size 16
// CHECK: SUMMARY: DeviceAddressSanitizer: global-buffer-overflow

// CHECK: ERROR: DeviceAddressSanitizer: global-buffer-overflow on address [[ADDR:0x[0-9a-f]+]]
// CHECK: #0 {{0x[0-9a-f]+}} in write_wide{{.*}}global-bounds.hip
// CHECK: [[ADDR]] is located 0 bytes after global variable 'Wide' {{.*}} of size 4096
// CHECK: SUMMARY: DeviceAddressSanitizer: global-buffer-overflow

// CHECK: ERROR: DeviceAddressSanitizer: global-buffer-overflow on address [[ADDR:0x[0-9a-f]+]]
// CHECK: #0 {{0x[0-9a-f]+}} in write_small{{.*}}global-bounds.hip
// CHECK: [[ADDR]] is located 32 bytes after global variable 'Small' {{.*}} of size 128
// CHECK: SUMMARY: DeviceAddressSanitizer: global-buffer-overflow

// CHECK: ERROR: DeviceAddressSanitizer: global-buffer-overflow on address [[ADDR:0x[0-9a-f]+]]
// CHECK: #0 {{0x[0-9a-f]+}} in write_big{{.*}}global-bounds.hip
// CHECK: [[ADDR]] is located 2048 bytes after global variable 'Big' {{.*}} of size 16384
// CHECK: SUMMARY: DeviceAddressSanitizer: global-buffer-overflow

// CHECK: ERROR: DeviceAddressSanitizer: global-buffer-overflow on address [[ADDR:0x[0-9a-f]+]]
// CHECK: #0 {{0x[0-9a-f]+}} in write_huge{{.*}}global-bounds.hip
// CHECK: [[ADDR]] is located 0 bytes after global variable 'Huge' {{.*}} of size 16777216
// CHECK: SUMMARY: DeviceAddressSanitizer: global-buffer-overflow
