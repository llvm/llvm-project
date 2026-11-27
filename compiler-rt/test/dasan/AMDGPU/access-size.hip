// RUN: %clang_hip_dasan %s -o %t %hip_libs
// RUN: %env_dasan_opts=halt_on_error=0 %run %t 2>&1 \
// RUN:   | FileCheck %s --implicit-check-not=SUMMARY
// RUN: %clang_hip_dasan -O1 %s -o %t %hip_libs
// RUN: %env_dasan_opts=halt_on_error=0 %run %t 2>&1 \
// RUN:   | FileCheck %s --implicit-check-not=SUMMARY

__global__ void write_byte(char *P) { P[256] = 1; }
__global__ void write_short(short *P) { P[128] = 1; }
__global__ void write_long(long *P) { P[32] = 1; }
__global__ void write_vector(float4 *P) { P[16] = float4(1.0f); }

__global__ void bump(int *P, int I) {
  __atomic_fetch_add(P + I, 1, __ATOMIC_RELAXED);
}

__global__ void claim(int *P, int I) {
  int Want = 0;
  __atomic_compare_exchange_n(P + I, &Want, 1, false, __ATOMIC_RELAXED,
                              __ATOMIC_RELAXED);
}

__global__ void copy_into(char *Dst, const char *Src, unsigned long N) {
  __builtin_memcpy(Dst + 128, Src, N);
}

__global__ void copy_out_of(char *Dst, const char *Src, unsigned long N) {
  __builtin_memcpy(Dst, Src + 128, N);
}

__global__ void fill(char *Dst, unsigned long N) {
  __builtin_memset(Dst + 128, 0x41, N);
}

int main() {
  char *P = nullptr, *Q = nullptr;
  CHECK_HIP(hipMalloc((void **)&P, 256));
  CHECK_HIP(hipMalloc((void **)&Q, 256));

  write_byte<<<1, 1>>>(P);
  write_short<<<1, 1>>>((short *)P);
  write_long<<<1, 1>>>((long *)P);
  write_vector<<<1, 1>>>((float4 *)P);
  bump<<<1, 1>>>((int *)P, 64);
  claim<<<1, 1>>>((int *)P, 64);
  copy_into<<<1, 1>>>(P, Q, 200);
  copy_out_of<<<1, 1>>>(P, Q, 200);
  fill<<<1, 1>>>(P, 200);
  CHECK_HIP(hipDeviceSynchronize());
  return 0;
}

// CHECK: WRITE of size 1 at
// CHECK: in write_byte{{.*}}access-size.hip
// CHECK: is located 0 bytes after 256-byte region
// CHECK: SUMMARY: DeviceAddressSanitizer: heap-buffer-overflow

// CHECK: WRITE of size 2 at
// CHECK: in write_short{{.*}}access-size.hip
// CHECK: is located 0 bytes after 256-byte region
// CHECK: SUMMARY: DeviceAddressSanitizer: heap-buffer-overflow

// CHECK: WRITE of size 8 at
// CHECK: in write_long{{.*}}access-size.hip
// CHECK: is located 0 bytes after 256-byte region
// CHECK: SUMMARY: DeviceAddressSanitizer: heap-buffer-overflow

// CHECK: WRITE of size 16 at
// CHECK: in write_vector{{.*}}access-size.hip
// CHECK: is located 0 bytes after 256-byte region
// CHECK: SUMMARY: DeviceAddressSanitizer: heap-buffer-overflow

// CHECK: WRITE of size 4 at
// CHECK: in bump{{.*}}access-size.hip
// CHECK: is located 0 bytes after 256-byte region
// CHECK: SUMMARY: DeviceAddressSanitizer: heap-buffer-overflow

// CHECK: WRITE of size 4 at
// CHECK: in claim{{.*}}access-size.hip
// CHECK: is located 0 bytes after 256-byte region
// CHECK: SUMMARY: DeviceAddressSanitizer: heap-buffer-overflow

// CHECK: WRITE of size 200 at
// CHECK: in copy_into{{.*}}access-size.hip
// CHECK: is located 128 bytes inside of 256-byte region
// CHECK: SUMMARY: DeviceAddressSanitizer: heap-buffer-overflow

// CHECK: READ of size 200 at
// CHECK: in copy_out_of{{.*}}access-size.hip
// CHECK: is located 128 bytes inside of 256-byte region
// CHECK: SUMMARY: DeviceAddressSanitizer: heap-buffer-overflow

// CHECK: WRITE of size 200 at
// CHECK: in fill{{.*}}access-size.hip
// CHECK: is located 128 bytes inside of 256-byte region
// CHECK: SUMMARY: DeviceAddressSanitizer: heap-buffer-overflow
