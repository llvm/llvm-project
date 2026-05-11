// RUN: %clangxx_asan -O0 -isystem %rocm_include %s -o %t -L%rocm_lib -lhsa-runtime64 \
// RUN:   -Wl,-rpath,%rocm_lib -Wl,-rpath,%compiler_rt_libdir
// RUN: not %run %t 2>&1 | FileCheck %s
//
// Regression test for the AddressSanitizer hsa_amd_memory_pool_allocate /
// hsa_amd_memory_pool_free interceptors: freeing the same pool allocation
// twice is diagnosed (same family of checks as double-free on malloc).
//
// REQUIRES: sanitizer-amdgpu, linux, stable-runtime, rocm
// UNSUPPORTED: android

#include "hsa_amd_test_helpers.h"

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <stdio.h>

int main() {
  if (hsa_amd_test_require_init())
    return 1;

  HsaAmdPoolSearch ps;
  if (hsa_amd_test_find_first_runtime_alloc_pool(&ps))
    return 1;

  void *mem = nullptr;
  if (hsa_amd_memory_pool_allocate(ps.pool, 64, 0, &mem) !=
          HSA_STATUS_SUCCESS ||
      !mem) {
    fprintf(stderr, "hsa_amd_memory_pool_allocate failed\n");
    return 1;
  }

  (void)hsa_amd_memory_pool_free(mem);
  (void)hsa_amd_memory_pool_free(mem);

  fprintf(stderr, "expected double-free report\n");
  return 0;
}

// CHECK: ERROR: AddressSanitizer: attempting double-free
// CHECK: SUMMARY: AddressSanitizer: double-free {{.*}}hsa_amd_memory_pool_allocate_double_free.cpp:36:{{[0-9]+}} in main
