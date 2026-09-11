// RUN: %clangxx_asan -O0 -isystem %rocm_include %s -o %t -L%rocm_lib -lhsa-runtime64 \
// RUN:   -Wl,-rpath,%rocm_lib -Wl,-rpath,%compiler_rt_libdir
// RUN: not %run %t 2>&1 | FileCheck %s
//
// Regression test for the AddressSanitizer hsa_amd_vmem_address_reserve_align /
// hsa_amd_vmem_address_free interceptors: Using the same freed reserved range twice is diagnosed as double-free.
//
// REQUIRES: linux, stable-runtime, rocm, hsa-vmem
// UNSUPPORTED: android

#include "hsa_amd_test_helpers.h"

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <stdio.h>

int main() {
  HSA_CHECK(hsa_init());

  // Size must be a non-zero multiple of the page size; address must be 0 so
  // the interceptor records the reservation for double-free diagnosis (see
  // asan_hsa_amd_vmem_address_reserve_align).
  const size_t kSize = 4096;
  void *mem = nullptr;

  // NOTE: To use `hipMallocManaged` way of reserving memory,
  // use `HSA_AMD_VMEM_ADDRESS_NO_REGISTER` in `flags`.
  HSA_CHECK(hsa_amd_vmem_address_reserve_align(&mem, kSize, /*address=*/0,
                                               /*alignment=*/4096,
                                               /*flags=*/0));

  (void)hsa_amd_vmem_address_free(mem, kSize);
  (void)hsa_amd_vmem_address_free(mem, kSize);

  fprintf(stderr, "expected double-free report\n");
  return 0;
}

// CHECK: ERROR: AddressSanitizer: attempting double-free on {{0x[0-9a-f]+}} in thread T0
// CHECK: is a device VMEM reservation of size {{[0-9]+}}
// CHECK: first freed by thread T0 here:
// CHECK: previously reserved by thread T0 here:
// CHECK: SUMMARY: AddressSanitizer: double-free
