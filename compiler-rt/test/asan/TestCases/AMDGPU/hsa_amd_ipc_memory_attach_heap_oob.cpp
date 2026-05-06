// RUN: %clangxx_asan -O0 -isystem %rocm_include %s -o %t -L%rocm_lib -lhsa-runtime64 \
// RUN:   -Wl,-rpath,%rocm_lib -Wl,-rpath,%compiler_rt_libdir
// RUN: not %run %t 2>&1 | FileCheck %s
//
// After hsa_amd_ipc_memory_attach, AddressSanitizer poisons a trailing redzone
// matching the pool allocation layout. A one-past-end store must be reported as
// a heap-buffer-overflow.
// hsa_amd_ipc_memory_create only supports coarse-grained GPU allocations; skip
// fine-grained pools and non-GPU agents.
//
// The bad store is instrumented host code; VRAM imports may need CPU access
// enabled (best-effort hsa_amd_agents_allow_access) or the fault can be SIGSEGV
// instead of AddressSanitizer.
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

static hsa_agent_t g_cpu_agent = {};

static hsa_status_t pick_first_cpu_agent(hsa_agent_t agent, void * /*data*/) {
  hsa_device_type_t dev = HSA_DEVICE_TYPE_GPU;
  if (hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &dev) !=
      HSA_STATUS_SUCCESS)
    return HSA_STATUS_SUCCESS;
  if (dev != HSA_DEVICE_TYPE_CPU)
    return HSA_STATUS_SUCCESS;
  g_cpu_agent = agent;
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

  g_cpu_agent.handle = 0;
  (void)hsa_iterate_agents(pick_first_cpu_agent, nullptr);
  if (g_cpu_agent.handle != 0) {
    /* Best-effort: allow the host CPU to access the imported range so the store
       below is a normal fault checked by ASan, not an unmapped-device SIGSEGV. */
    (void)hsa_amd_agents_allow_access(/*num_agents=*/1, &g_cpu_agent,
                                      /*flags=*/nullptr, mapped);
  }

  auto *p = reinterpret_cast<volatile char *>(mapped);
  // One byte past the 64-byte imported region; should land in ASan's tail redzone.
  p[kBytes] = 1;

  fprintf(stderr, "expected heap-buffer-overflow after ipc attach\n");
  (void)hsa_amd_ipc_memory_detach(mapped);
  (void)hsa_amd_memory_pool_free(mem);
  return 0;
}

// CHECK: ERROR: AddressSanitizer: heap-buffer-overflow
// CHECK-NEXT: WRITE of size 1 at {{0x[0-9a-f]+}} thread T0
// CHECK: SUMMARY: AddressSanitizer: heap-buffer-overflow {{.*}}hsa_amd_ipc_memory_attach_heap_oob.cpp:149:{{[0-9]+}} in main
