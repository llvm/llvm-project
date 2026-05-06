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
// REQUIRES: sanitizer-amdgpu, linux, stable-runtime, rocm
// UNSUPPORTED: android

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <stdio.h>

struct PoolSearch {
  hsa_amd_memory_pool_t pool;
  bool found;
};

static hsa_status_t find_coarse_gpu_ipc_pool(hsa_amd_memory_pool_t pool,
                                             void *data) {
  auto *ps = static_cast<PoolSearch *>(data);

  hsa_amd_segment_t segment = HSA_AMD_SEGMENT_PRIVATE;
  if (hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT,
                                   &segment) != HSA_STATUS_SUCCESS)
    return HSA_STATUS_SUCCESS;
  if (segment != HSA_AMD_SEGMENT_GLOBAL)
    return HSA_STATUS_SUCCESS;

  uint32_t global_flags = 0;
  if (hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS,
                                   &global_flags) != HSA_STATUS_SUCCESS)
    return HSA_STATUS_SUCCESS;
  if ((global_flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED) == 0)
    return HSA_STATUS_SUCCESS;

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

static hsa_status_t find_gpu_agent_with_ipc_pool(hsa_agent_t agent,
                                                 void *data) {
  hsa_device_type_t dev = HSA_DEVICE_TYPE_CPU;
  if (hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &dev) !=
      HSA_STATUS_SUCCESS)
    return HSA_STATUS_SUCCESS;
  if (dev != HSA_DEVICE_TYPE_GPU)
    return HSA_STATUS_SUCCESS;

  auto *ps = static_cast<PoolSearch *>(data);
  ps->found = false;
  hsa_status_t st =
      hsa_amd_agent_iterate_memory_pools(agent, find_coarse_gpu_ipc_pool, ps);
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

  hsa_status_t it = hsa_iterate_agents(find_gpu_agent_with_ipc_pool, &ps);
  if (it != HSA_STATUS_SUCCESS && it != HSA_STATUS_INFO_BREAK) {
    fprintf(stderr, "hsa_iterate_agents failed\n");
    return 1;
  }
  if (!ps.found) {
    fprintf(stderr,
            "no coarse-grained GPU runtime-alloc HSA memory pool found\n");
    return 1;
  }

  constexpr size_t kBytes = 64;
  void *mem = nullptr;
  if (hsa_amd_memory_pool_allocate(ps.pool, kBytes, 0, &mem) !=
          HSA_STATUS_SUCCESS ||
      !mem) {
    fprintf(stderr, "hsa_amd_memory_pool_allocate failed\n");
    return 1;
  }

  hsa_amd_ipc_memory_t ipc = {};
  if (hsa_amd_ipc_memory_create(mem, kBytes, &ipc) != HSA_STATUS_SUCCESS) {
    fprintf(stderr, "hsa_amd_ipc_memory_create failed\n");
    (void)hsa_amd_memory_pool_free(mem);
    return 1;
  }

  void *mapped = nullptr;
  if (hsa_amd_ipc_memory_attach(&ipc, kBytes, /*num_agents=*/0,
                                /*mapping_agents=*/nullptr,
                                &mapped) != HSA_STATUS_SUCCESS ||
      !mapped) {
    fprintf(stderr, "hsa_amd_ipc_memory_attach failed\n");
    (void)hsa_amd_memory_pool_free(mem);
    return 1;
  }

  hsa_amd_pointer_info_t info = {};
  info.size = sizeof(hsa_amd_pointer_info_t);
  if (hsa_amd_pointer_info(mapped, &info, nullptr, nullptr, nullptr) !=
      HSA_STATUS_SUCCESS) {
    fprintf(stderr, "hsa_amd_pointer_info on imported mapping failed\n");
    (void)hsa_amd_ipc_memory_detach(mapped);
    (void)hsa_amd_memory_pool_free(mem);
    return 1;
  }

  if (hsa_amd_ipc_memory_detach(mapped) != HSA_STATUS_SUCCESS) {
    fprintf(stderr, "hsa_amd_ipc_memory_detach failed\n");
    (void)hsa_amd_memory_pool_free(mem);
    return 1;
  }

  if (hsa_amd_memory_pool_free(mem) != HSA_STATUS_SUCCESS) {
    fprintf(stderr, "hsa_amd_memory_pool_free failed\n");
    return 1;
  }

  printf("ipc roundtrip ok\n");
  return 0;
}

// CHECK: ipc roundtrip ok
