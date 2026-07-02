// RUN: %clangxx_asan -O0 -isystem %rocm_include %s -o %t -L%rocm_lib -lhsa-runtime64 \
// RUN:   -Wl,-rpath,%rocm_lib -Wl,-rpath,%compiler_rt_libdir
// RUN: not %run %t 2>&1 | FileCheck %s
//
// Regression test for the AddressSanitizer hsa_amd_vmem_address_reserve_align /
// hsa_amd_vmem_address_free interceptors: Using the same freed reserved range is diagnosed as use-after-free.
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

  // Size must be a non-zero multiple of the page size; address must be 0 so
  // the interceptor records the reservation for double-free diagnosis (see
  // asan_hsa_amd_vmem_address_reserve_align).
  const size_t kSize = 4096;
  void *mem = nullptr;

  // NOTE: To use `hipMallocManaged` way of reserving memory,
  // use `HSA_AMD_VMEM_ADDRESS_NO_REGISTER` in `flags`.
  if (hsa_amd_vmem_address_reserve_align(
          &mem, kSize, /*address=*/0,
          /*alignment=*/4096,
          /*flags=*/HSA_AMD_VMEM_ADDRESS_NO_REGISTER) != HSA_STATUS_SUCCESS ||
      !mem) {
    fprintf(stderr, "hsa_amd_vmem_address_reserve_align failed\n");
    return 1;
  }

  (void)hsa_amd_vmem_address_free(mem, kSize);
  auto *p = reinterpret_cast<volatile char *>(mem);
  p[0] = 1; // Use-after-free

  fprintf(stderr, "expected double-free report\n");
  return 0;
}

// CHECK: ERROR: AddressSanitizer: heap-use-after-free
// CHECK-NEXT: WRITE of size 1 at {{0x[0-9a-f]+}} thread T0
// CHECK: SUMMARY: AddressSanitizer: heap-use-after-free {{.*}}hsa_amd_vmem_address_reserve_align_uaf.cpp:41:{{[0-9]+}} in main