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

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <stdio.h>

struct PoolSearch {
  hsa_amd_memory_pool_t pool;
  bool found;
};

static hsa_status_t find_alloc_pool(hsa_amd_memory_pool_t pool, void *data) {
  auto *ps = static_cast<PoolSearch *>(data);
  bool allow = false;
  if (hsa_amd_memory_pool_get_info(
          pool, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED, &allow) !=
          HSA_STATUS_SUCCESS ||
      !allow)
    return HSA_STATUS_SUCCESS;
  ps->pool = pool;
  ps->found = true;
  return HSA_STATUS_INFO_BREAK;
}

static hsa_status_t find_agent_with_pool(hsa_agent_t agent, void *data) {
  (void)agent;
  auto *ps = static_cast<PoolSearch *>(data);
  ps->found = false;
  hsa_status_t st =
      hsa_amd_agent_iterate_memory_pools(agent, find_alloc_pool, ps);
  if (st != HSA_STATUS_SUCCESS && st != HSA_STATUS_INFO_BREAK)
    return st;
  if (ps->found)
    return HSA_STATUS_INFO_BREAK;
  return HSA_STATUS_SUCCESS;
}

int main() {
  if (hsa_init() != HSA_STATUS_SUCCESS) {
    fprintf(stderr, "hsa_init failed\n");
    return 1;
  }

  PoolSearch ps = {};
  ps.pool.handle = 0;
  ps.found = false;

  hsa_status_t it = hsa_iterate_agents(find_agent_with_pool, &ps);
  if (it != HSA_STATUS_SUCCESS && it != HSA_STATUS_INFO_BREAK) {
    fprintf(stderr, "hsa_iterate_agents failed\n");
    return 1;
  }
  if (!ps.found) {
    fprintf(stderr, "no runtime-alloc HSA memory pool found\n");
    return 1;
  }

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
// CHECK: SUMMARY: AddressSanitizer: double-free {{.*}}hsa_amd_memory_pool_allocate_double_free.cpp:77:{{[0-9]+}} in main
