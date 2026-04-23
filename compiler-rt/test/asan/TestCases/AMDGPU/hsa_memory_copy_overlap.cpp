// RUN: %clangxx_asan -O0 -isystem %rocm_include %s -o %t -L%rocm_lib -lhsa-runtime64 \
// RUN:   -Wl,-rpath,%rocm_lib -Wl,-rpath,%compiler_rt_libdir
// RUN: not %run %t 2>&1 | FileCheck %s
//
// Regression test for the AddressSanitizer hsa_memory_copy interceptor: invalid
// overlapping ranges are diagnosed (same family of checks as memcpy).
//
// REQUIRES: sanitizer-amdgpu, linux, stable-runtime, rocm
// UNSUPPORTED: android

#include <hsa/hsa.h>

#include <stdio.h>
#include <stdlib.h>

int main() {
  if (hsa_init() != HSA_STATUS_SUCCESS) {
    fprintf(stderr, "hsa_init failed\n");
    return 1;
  }

  char buf[128];
  char *dst = buf;
  char *src = buf + 40;
  // Ranges [buf, buf+64) and [buf+40, buf+104) overlap; dst != src so the
  // interceptor runs CHECK_RANGES_OVERLAP.
  (void)hsa_memory_copy(dst, src, 64);
  fprintf(stderr, "expected hsa_memory_copy overlap report\n");
  return 0;
}

// CHECK: hsa_memory_copy-param-overlap: memory ranges
// CHECK: [{{0x.*,[ ]*0x.*}}) and [{{0x.*,[ ]*0x.*}}) overlap
// CHECK: SUMMARY: AddressSanitizer: hsa_memory_copy-param-overlap
