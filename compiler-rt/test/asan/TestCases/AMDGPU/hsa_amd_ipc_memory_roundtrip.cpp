// RUN: %clangxx_asan -O0 -isystem %rocm_include %s -o %t -L%rocm_lib -lhsa-runtime64 \
// RUN:   -Wl,-rpath,%rocm_lib -Wl,-rpath,%compiler_rt_libdir
// RUN: %run %t 2>&1 | FileCheck %s
//
// Regression test for AddressSanitizer hsa_amd_ipc_memory_{create,attach,detach}:
// same-process IPC round-trip on a pool allocation (unwrap on create, user pointer
// adjustment and shadow on attach, base adjustment on detach).
// hsa_amd_ipc_memory_create only supports coarse-grained GPU allocations; skip
// fine-grained pools and non-GPU agents.
//
// Coarse-grained device memory is often not mapped for CPU stores; do not
// read/write *mapped from the host. Validate with hsa_amd_pointer_info instead.
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
  if (hsa_amd_test_find_first_coarse_gpu_ipc_pool(&ps))
    return 1;

  constexpr size_t kBytes = 64;
  void *mem = nullptr;
  HSA_CHECK(hsa_amd_memory_pool_allocate(ps.pool, kBytes, 0, &mem));

  hsa_amd_ipc_memory_t ipc = {};
  HSA_CHECK(hsa_amd_ipc_memory_create(mem, kBytes, &ipc));

  void *mapped = nullptr;
  HSA_CHECK(hsa_amd_ipc_memory_attach(&ipc, kBytes, /*num_agents=*/0,
                                      /*mapping_agents=*/nullptr, &mapped));

  hsa_amd_pointer_info_t info = {};
  info.size = sizeof(hsa_amd_pointer_info_t);
  HSA_CHECK(hsa_amd_pointer_info(mapped, &info, nullptr, nullptr, nullptr));

  HSA_CHECK(hsa_amd_ipc_memory_detach(mapped));

  HSA_CHECK(hsa_amd_memory_pool_free(mem));

  printf("ipc roundtrip ok\n");
  return 0;
}

// CHECK: ipc roundtrip ok
