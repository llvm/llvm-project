// RUN: %clang_hip_dasan %s -o %t %hip_libs
// RUN: %env_dasan_opts=halt_on_error=0 %run %t 2>&1 \
// RUN:   | FileCheck %s --implicit-check-not=SUMMARY
// RUN: %clang_hip_dasan -O1 %s -o %t %hip_libs
// RUN: %env_dasan_opts=halt_on_error=0 %run %t 2>&1 \
// RUN:   | FileCheck %s --implicit-check-not=SUMMARY

template <int> __global__ void touch(char *P, long I) { P[I] = 1; }

static const long Sizes[] = {16, 1000, 4096, 100000};

template <int Id> static void Touch(char *P, long N) {
  touch<Id><<<1, 1>>>(P, N - 1);
  touch<Id><<<1, 1>>>(P, N);
}

int main() {
  int Id = 0;
  for (long N : Sizes) {
    char *P = nullptr;
    CHECK_HIP(hipMalloc((void **)&P, N));
    if ((unsigned long)P % 256)
      return 1;
    if (Id == 0)
      Touch<0>(P, N);
    else if (Id == 1)
      Touch<1>(P, N);
    else if (Id == 2)
      Touch<2>(P, N);
    else
      Touch<3>(P, N);
    ++Id;
  }
  CHECK_HIP(hipDeviceSynchronize());
  return 0;
}

// CHECK: #0 {{0x[0-9a-f]+}} in {{.*}}touch{{.*}}size-classes.hip
// CHECK: is located 0 bytes after 16-byte region
// CHECK: SUMMARY: DeviceAddressSanitizer: heap-buffer-overflow
// CHECK: #0 {{0x[0-9a-f]+}} in {{.*}}touch{{.*}}size-classes.hip
// CHECK: is located 0 bytes after 1000-byte region
// CHECK: SUMMARY: DeviceAddressSanitizer: heap-buffer-overflow
// CHECK: #0 {{0x[0-9a-f]+}} in {{.*}}touch{{.*}}size-classes.hip
// CHECK: is located 0 bytes after 4096-byte region
// CHECK: SUMMARY: DeviceAddressSanitizer: heap-buffer-overflow
// CHECK: #0 {{0x[0-9a-f]+}} in {{.*}}touch{{.*}}size-classes.hip
// CHECK: is located 0 bytes after 100000-byte region
// CHECK: SUMMARY: DeviceAddressSanitizer: heap-buffer-overflow
