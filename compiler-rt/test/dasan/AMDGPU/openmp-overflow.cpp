// RUN: %clang_omp_offload_dasan %s -o %t
// RUN: %run %t clean
// RUN: not %run %t heap 2>&1 | FileCheck %s --check-prefix=HEAP
// RUN: %if dasan-globals %{ %env_dasan_opts=halt_on_error=0 %run %t heap global 2>&1 | FileCheck %s --check-prefixes=HEAP,GLOBAL --implicit-check-not=SUMMARY %} %else %{ %env_dasan_opts=halt_on_error=0 %run %t heap 2>&1 | FileCheck %s --check-prefix=HEAP %}
// RUN: %clang_omp_offload_dasan -O1 %s -o %t
// RUN: %run %t clean
// RUN: not %run %t heap 2>&1 | FileCheck %s --check-prefix=HEAP
// RUN: %if dasan-globals %{ %env_dasan_opts=halt_on_error=0 %run %t heap global 2>&1 | FileCheck %s --check-prefixes=HEAP,GLOBAL --implicit-check-not=SUMMARY %} %else %{ %env_dasan_opts=halt_on_error=0 %run %t heap 2>&1 | FileCheck %s --check-prefix=HEAP %}
// RUN: %clang_omp_offload_dasan -shared-libsan %s -o %t
// RUN: not %run %t heap 2>&1 | FileCheck %s --check-prefix=HEAP

#include <omp.h>

#pragma omp begin declare target
int Table[16];
#pragma omp end declare target

int main(int argc, char **argv) {
  int Dev = omp_get_default_device();
  int *P = (int *)omp_target_alloc(64 * sizeof(int), Dev);
  if (!P)
    return 2;

  for (int A = 1; A < argc; ++A) {
    int Past = argv[A][0] != 'c';
    if (argv[A][0] != 'g') {
#pragma omp target is_device_ptr(P) firstprivate(Past) device(Dev)
      P[63 + Past] = 1;
    }
    if (argv[A][0] != 'h') {
#pragma omp target firstprivate(Past) device(Dev)
      Table[15 + Past] = 1;
    }
  }
  omp_target_free(P, Dev);
  return 0;
}

// HEAP: ERROR: DeviceAddressSanitizer: heap-buffer-overflow on address [[ADDR:0x[0-9a-f]+]]
// HEAP: WRITE of size 4 at [[ADDR]]
// HEAP: #0 {{0x[0-9a-f]+}} in {{.*}}openmp-overflow.cpp
// HEAP: [[ADDR]] is located 0 bytes after 256-byte region
// HEAP: SUMMARY: DeviceAddressSanitizer: heap-buffer-overflow

// GLOBAL: ERROR: DeviceAddressSanitizer: global-buffer-overflow on address [[ADDR:0x[0-9a-f]+]]
// GLOBAL: WRITE of size 4 at [[ADDR]]
// GLOBAL: #0 {{0x[0-9a-f]+}} in {{.*}}openmp-overflow.cpp
// GLOBAL: [[ADDR]] is located 0 bytes after global variable 'Table' {{.*}} of size 64
// GLOBAL: SUMMARY: DeviceAddressSanitizer: global-buffer-overflow
