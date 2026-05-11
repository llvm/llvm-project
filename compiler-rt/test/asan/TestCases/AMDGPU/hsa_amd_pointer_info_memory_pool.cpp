// RUN: %clangxx_asan -O0 -isystem %rocm_include %s -o %t -L%rocm_lib -lhsa-runtime64 \
// RUN:   -Wl,-rpath,%rocm_lib -Wl,-rpath,%compiler_rt_libdir
// RUN: %run %t 2>&1 | FileCheck %s
//
// Regression test for the AddressSanitizer hsa_amd_pointer_info interceptor on
// hsa_amd_memory_pool_allocate pointers: reported sizeInBytes matches the user
// request (ASan unwraps the page-sized host wrapper from pointer metadata).
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

  hsa_amd_pointer_info_t info = {};
  info.size = sizeof(hsa_amd_pointer_info_t);

  if (hsa_amd_pointer_info(mem, &info, nullptr, nullptr, nullptr) !=
      HSA_STATUS_SUCCESS) {
    fprintf(stderr, "hsa_amd_pointer_info failed\n");
    return 1;
  }

  printf("pointer_info_pool type: %d\n", info.type);
  printf("pointer_info_pool sizeInBytes: %zu\n", info.sizeInBytes);
  printf("pointer_info_pool begin: %p\n", info.agentBaseAddress);
  printf("pointer_info_pool end: %p\n",
         (void *)((uintptr_t)info.agentBaseAddress + info.sizeInBytes));

  if (hsa_amd_memory_pool_free(mem) != HSA_STATUS_SUCCESS) {
    fprintf(stderr, "hsa_amd_memory_pool_free failed\n");
    return 1;
  }
  return 0;
}

// CHECK: pointer_info_pool type: 1
// CHECK-NEXT: pointer_info_pool sizeInBytes: 64
// CHECK-NEXT: pointer_info_pool begin: 0x{{[0-9a-f]+}}
// CHECK-NEXT: pointer_info_pool end: 0x{{[0-9a-f]+}}
