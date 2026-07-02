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

#include "hsa_amd_test_helpers.h"

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <stdio.h>

int main() {
  if (hsa_amd_test_require_init())
    return 1;

  HsaAmdPoolSearch ps;
  if (hsa_amd_test_find_first_coarse_gpu_ipc_pool(&ps))
    return 1;

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

  HsaAmdCpuAgentPick cpu;
  hsa_amd_test_cpu_agent_pick_init(&cpu);
  (void)hsa_iterate_agents(hsa_amd_test_pick_first_cpu_agent_cb, &cpu);
  if (cpu.agent.handle != 0) {
    /* Best-effort: allow the host CPU to access the imported range so the store
       below is a normal fault checked by ASan, not an unmapped-device SIGSEGV. */
    (void)hsa_amd_agents_allow_access(/*num_agents=*/1, &cpu.agent,
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
// CHECK: SUMMARY: AddressSanitizer: heap-buffer-overflow {{.*}}hsa_amd_ipc_memory_attach_heap_oob.cpp:71:{{[0-9]+}} in main
