// RUN: %clangxx_asan -O0 -isystem %rocm_include %s -o %t -L%rocm_lib -lhsa-runtime64 \
// RUN:   -Wl,-rpath,%rocm_lib -Wl,-rpath,%compiler_rt_libdir
// RUN: %run %t 2>&1 | FileCheck %s
//
// Regression test for the AddressSanitizer hsa_amd_pointer_info interceptor on
// freed pointers: quarantine keeps the ROCr block mapped so that
// use-after-free stays detectable, which means ROCr still describes a freed
// allocation as live HSA memory. ASan must report it as unowned instead.
//
// REQUIRES: linux, stable-runtime, rocm
// UNSUPPORTED: android

#include "hsa_amd_test_helpers.h"

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <stdio.h>

int main() {
  HSA_CHECK(hsa_init());

  HsaAmdPoolSearch ps;
  if (hsa_amd_test_find_first_runtime_alloc_pool(&ps))
    return 1;

  void *mem = nullptr;
  HSA_CHECK(hsa_amd_memory_pool_allocate(ps.pool, 64, 0, &mem));
  HSA_CHECK(hsa_amd_memory_pool_free(mem));

  hsa_amd_pointer_info_t info = {};
  info.size = sizeof(hsa_amd_pointer_info_t);
  info.type = HSA_EXT_POINTER_TYPE_HSA;
  info.agentBaseAddress = mem;
  info.hostBaseAddress = mem;
  info.sizeInBytes = 64;

  HSA_CHECK(hsa_amd_pointer_info(mem, &info, nullptr, nullptr, nullptr));

  printf("freed type: %d\n", info.type);
  printf("freed sizeInBytes: %zu\n", info.sizeInBytes);
  printf("freed agent base null: %d\n", info.agentBaseAddress == nullptr);
  printf("freed host base null: %d\n", info.hostBaseAddress == nullptr);
  return 0;
}

// CHECK: freed type: 0
// CHECK-NEXT: freed sizeInBytes: 0
// CHECK-NEXT: freed agent base null: 1
// CHECK-NEXT: freed host base null: 1
