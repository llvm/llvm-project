// RUN: %clang_hip_dasan %s -o %t %hip_libs
// RUN: %env_dasan_opts=alias_globals=0 %run %t global
// RUN: %env_dasan_opts=halt_on_error=0 %run %t heap 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CARRYON
// RUN: %env_dasan_opts=print_stats=1 %run %t 2>&1 \
// RUN:   | FileCheck %s --check-prefix=STATS
// RUN: %clang_hip_dasan -O1 %s -o %t %hip_libs
// RUN: %env_dasan_opts=alias_globals=0 %run %t global
// RUN: %env_dasan_opts=halt_on_error=0 %run %t heap 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CARRYON
// RUN: %env_dasan_opts=print_stats=1 %run %t 2>&1 \
// RUN:   | FileCheck %s --check-prefix=STATS

#include <stdio.h>

__device__ int Table[16];

__global__ void write_heap(int *P) { P[64] = 1; }

__global__ void write_global(int I) { Table[I] = 1; }

int main(int argc, char **argv) {
  const char Mode = argc > 1 ? argv[1][0] : 0;
  int *P = nullptr;
  CHECK_HIP(hipMalloc((void **)&P, 64 * sizeof(int)));

  if (Mode == 'h')
    write_heap<<<1, 1>>>(P);
  else if (Mode == 'g')
    write_global<<<1, 1>>>(16);
  CHECK_HIP(hipDeviceSynchronize());
  fprintf(stderr, "synchronized\n");
  return 0;
}

// The wave is already gone. The host carries on, and says so once.
// CARRYON: WARNING: DeviceAddressSanitizer: halt_on_error=0 cannot resume
// CARRYON: ERROR: DeviceAddressSanitizer: heap-buffer-overflow
// CARRYON: SUMMARY: DeviceAddressSanitizer: heap-buffer-overflow
// CARRYON-NOT: HINT
// CARRYON: synchronized

// STATS: stats: {{[0-9]+}} placed ({{[0-9]+}} above their own class), {{[0-9]+}} freed
// STATS: stats: {{[0-9]+}} code objects placed over {{[0-9]+}} chunks
