// RUN: %clangxx_asan -O0 -isystem %rocm_include %s -o %t -L%rocm_lib -lhsa-runtime64 \
// RUN:   -Wl,-rpath,%rocm_lib -Wl,-rpath,%compiler_rt_libdir
// RUN: not %run %t 2>&1 | FileCheck %s
//
// Regression test for the AddressSanitizer hsa_amd_vmem_address_reserve_align /
// hsa_amd_vmem_address_free interceptors: freeing the same reserved range twice
// is diagnosed.
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
  // use either `HSA_AMD_VMEM_ADDRESS_NO_REGISTER` in flags.
  if (hsa_amd_vmem_address_reserve_align(
          &mem, kSize, /*address=*/0,
          /*alignment=*/4096,
          /*flags=*/HSA_AMD_VMEM_ADDRESS_NO_REGISTER) != HSA_STATUS_SUCCESS ||
      !mem) {
    fprintf(stderr, "hsa_amd_vmem_address_reserve_align failed\n");
    return 1;
  }

  (void)hsa_amd_vmem_address_free(mem, kSize);
  (void)hsa_amd_vmem_address_free(mem, kSize);

  fprintf(stderr, "expected double-free report\n");
  return 0;
}

// CHECK: ERROR: AddressSanitizer: attempting double-free
// CHECK: SUMMARY: AddressSanitizer: double-free
