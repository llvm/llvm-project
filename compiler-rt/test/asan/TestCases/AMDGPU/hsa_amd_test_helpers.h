//===-- hsa_amd_test_helpers.h - shared helpers for AMDGPU ASan HSA tests --===//
//
// Common ROCm/HSA discovery and init helpers for hsa_amd*.cpp tests in this
// directory. Intended only for compiler-rt lit tests.
//
//===----------------------------------------------------------------------===//

#ifndef COMPILER_RT_ASAN_AMDGPU_HSA_AMD_TEST_HELPERS_H
#define COMPILER_RT_ASAN_AMDGPU_HSA_AMD_TEST_HELPERS_H

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <stdio.h>

inline int hsa_amd_test_require_init() {
  if (hsa_init() != HSA_STATUS_SUCCESS) {
    fprintf(stderr, "hsa_init failed\n");
    return 1;
  }
  return 0;
}

/// Return 1 if `st` indicates a hard failure from hsa_iterate_agents.
inline int hsa_amd_test_iterate_agents_ok(hsa_status_t st) {
  if (st != HSA_STATUS_SUCCESS && st != HSA_STATUS_INFO_BREAK) {
    fprintf(stderr, "hsa_iterate_agents failed\n");
    return 1;
  }
  return 0;
}

struct HsaAmdPoolSearch {
  hsa_amd_memory_pool_t pool;
  bool found;
};

inline hsa_status_t
hsa_amd_test_find_runtime_alloc_pool_cb(hsa_amd_memory_pool_t pool,
                                        void *data) {
  auto *ps = static_cast<HsaAmdPoolSearch *>(data);
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

inline hsa_status_t
hsa_amd_test_find_agent_with_runtime_alloc_pool_cb(hsa_agent_t agent,
                                                   void *data) {
  (void)agent;
  auto *ps = static_cast<HsaAmdPoolSearch *>(data);
  ps->found = false;
  hsa_status_t st = hsa_amd_agent_iterate_memory_pools(
      agent, hsa_amd_test_find_runtime_alloc_pool_cb, ps);
  if (st != HSA_STATUS_SUCCESS && st != HSA_STATUS_INFO_BREAK)
    return st;
  if (ps->found)
    return HSA_STATUS_INFO_BREAK;
  return HSA_STATUS_SUCCESS;
}

inline int hsa_amd_test_find_first_runtime_alloc_pool(HsaAmdPoolSearch *ps) {
  ps->pool.handle = 0;
  ps->found = false;
  hsa_status_t it = hsa_iterate_agents(
      hsa_amd_test_find_agent_with_runtime_alloc_pool_cb, ps);
  if (hsa_amd_test_iterate_agents_ok(it))
    return 1;
  if (!ps->found) {
    fprintf(stderr, "no runtime-alloc HSA memory pool found\n");
    return 1;
  }
  return 0;
}

inline hsa_status_t
hsa_amd_test_find_coarse_gpu_ipc_pool_cb(hsa_amd_memory_pool_t pool,
                                         void *data) {
  auto *ps = static_cast<HsaAmdPoolSearch *>(data);

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

inline hsa_status_t
hsa_amd_test_find_gpu_agent_with_ipc_pool_cb(hsa_agent_t agent, void *data) {
  hsa_device_type_t dev = HSA_DEVICE_TYPE_CPU;
  if (hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &dev) !=
      HSA_STATUS_SUCCESS)
    return HSA_STATUS_SUCCESS;
  if (dev != HSA_DEVICE_TYPE_GPU)
    return HSA_STATUS_SUCCESS;

  auto *ps = static_cast<HsaAmdPoolSearch *>(data);
  ps->found = false;
  hsa_status_t st = hsa_amd_agent_iterate_memory_pools(
      agent, hsa_amd_test_find_coarse_gpu_ipc_pool_cb, ps);
  if (st != HSA_STATUS_SUCCESS && st != HSA_STATUS_INFO_BREAK)
    return st;
  if (ps->found)
    return HSA_STATUS_INFO_BREAK;
  return HSA_STATUS_SUCCESS;
}

inline int hsa_amd_test_find_first_coarse_gpu_ipc_pool(HsaAmdPoolSearch *ps) {
  ps->pool.handle = 0;
  ps->found = false;
  hsa_status_t it =
      hsa_iterate_agents(hsa_amd_test_find_gpu_agent_with_ipc_pool_cb, ps);
  if (hsa_amd_test_iterate_agents_ok(it))
    return 1;
  if (!ps->found) {
    fprintf(stderr,
            "no coarse-grained GPU runtime-alloc HSA memory pool found\n");
    return 1;
  }
  return 0;
}

struct HsaAmdCpuAgentPick {
  hsa_agent_t agent;
};

inline void hsa_amd_test_cpu_agent_pick_init(HsaAmdCpuAgentPick *out) {
  out->agent.handle = 0;
}

inline hsa_status_t hsa_amd_test_pick_first_cpu_agent_cb(hsa_agent_t agent,
                                                         void *data) {
  auto *out = static_cast<HsaAmdCpuAgentPick *>(data);
  hsa_device_type_t dev = HSA_DEVICE_TYPE_GPU;
  if (hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &dev) !=
      HSA_STATUS_SUCCESS)
    return HSA_STATUS_SUCCESS;
  if (dev != HSA_DEVICE_TYPE_CPU)
    return HSA_STATUS_SUCCESS;
  out->agent = agent;
  return HSA_STATUS_INFO_BREAK;
}

struct HsaAmdAgentPick {
  hsa_agent_t agent;
};

inline void hsa_amd_test_agent_pick_init(HsaAmdAgentPick *out) {
  out->agent.handle = 0;
}

inline hsa_status_t hsa_amd_test_pick_first_agent_cb(hsa_agent_t agent,
                                                     void *data) {
  auto *out = static_cast<HsaAmdAgentPick *>(data);
  out->agent = agent;
  return HSA_STATUS_INFO_BREAK;
}

#endif // COMPILER_RT_ASAN_AMDGPU_HSA_AMD_TEST_HELPERS_H
