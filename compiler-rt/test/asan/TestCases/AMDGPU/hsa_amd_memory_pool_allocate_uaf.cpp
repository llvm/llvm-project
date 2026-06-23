// RUN: %clangxx_asan -O0 -isystem %rocm_include %s -o %t -L%rocm_lib -lhsa-runtime64 \
// RUN:   -Wl,-rpath,%rocm_lib -Wl,-rpath,%compiler_rt_libdir
// RUN: not %run %t 2>&1 | FileCheck %s
//
// Regression test for the AddressSanitizer hsa_amd_memory_pool_allocate /
// hsa_amd_memory_pool_free interceptors: Using the same freed pool allocation twice is diagnosed as use-after-free.
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

  constexpr size_t kBytes = 64;
  void *mem = nullptr;
  if (hsa_amd_memory_pool_allocate(ps.pool, kBytes, 0, &mem) !=
          HSA_STATUS_SUCCESS ||
      !mem) {
    fprintf(stderr, "hsa_amd_memory_pool_allocate failed\n");
    return 1;
  }

  (void)hsa_amd_memory_pool_free(mem);
  auto *p = reinterpret_cast<volatile char *>(mem);
  p[0] = 1; // Use-after-free

  fprintf(stderr, "expected use-after-free report\n");
  return 0;
}

// CHECK: ERROR: AddressSanitizer: heap-use-after-free
// CHECK-NEXT: WRITE of size 1 at {{0x[0-9a-f]+}} thread T0
// CHECK: SUMMARY: AddressSanitizer: heap-use-after-free {{.*}}hsa_amd_memory_pool_allocate_uaf.cpp:37:{{[0-9]+}} in main
