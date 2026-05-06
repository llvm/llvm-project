// RUN: %clangxx_asan -O0 -isystem %rocm_include %s -o %t -L%rocm_lib -lhsa-runtime64 \
// RUN:   -Wl,-rpath,%rocm_lib -Wl,-rpath,%compiler_rt_libdir
// RUN: not %run %t 2>&1 | FileCheck %s
//
// Regression test for the AddressSanitizer hsa_amd_memory_async_copy interceptor:
// invalid overlapping ranges are diagnosed (same family of checks as memcpy).
//
// REQUIRES: sanitizer-amdgpu, linux, stable-runtime, rocm
// UNSUPPORTED: android

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <stdio.h>
#include <stdlib.h>

static hsa_agent_t g_agent = {};

static hsa_status_t pick_first_agent(hsa_agent_t agent, void * /*data*/) {
  g_agent = agent;
  return HSA_STATUS_INFO_BREAK;
}

int main() {
  if (hsa_init() != HSA_STATUS_SUCCESS) {
    fprintf(stderr, "hsa_init failed\n");
    return 1;
  }

  hsa_status_t it = hsa_iterate_agents(pick_first_agent, nullptr);
  if (it != HSA_STATUS_SUCCESS && it != HSA_STATUS_INFO_BREAK) {
    fprintf(stderr, "hsa_iterate_agents failed\n");
    return 1;
  }
  if (g_agent.handle == 0) {
    fprintf(stderr, "no HSA agent found\n");
    return 1;
  }

  hsa_signal_t completion = {};
  if (hsa_signal_create(/*initial_value=*/0, /*num_consumers=*/0,
                        /*consumers=*/nullptr,
                        &completion) != HSA_STATUS_SUCCESS) {
    fprintf(stderr, "hsa_signal_create failed\n");
    return 1;
  }

  char buf[128];
  char *dst = buf;
  char *src = buf + 40;
  // Ranges [buf, buf+64) and [buf+40, buf+104) overlap; dst != src so the
  // interceptor runs CHECK_RANGES_OVERLAP before scheduling the async copy.
  (void)hsa_amd_memory_async_copy(dst, g_agent, src, g_agent, 64,
                                  /*num_dep_signals=*/0,
                                  /*dep_signals=*/nullptr, completion);
  fprintf(stderr, "expected hsa_amd_memory_async_copy overlap report\n");
  return 0;
}

// CHECK: hsa_amd_memory_async_copy-param-overlap: memory ranges
// CHECK: [{{0x.*,[ ]*0x.*}}) and [{{0x.*,[ ]*0x.*}}) overlap
// CHECK: SUMMARY: AddressSanitizer: hsa_amd_memory_async_copy-param-overlap
