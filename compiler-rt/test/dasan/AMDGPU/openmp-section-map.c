// RUN: %clang_omp_offload_dasan -O2 %s -o %t
// RUN: %run %t
// RUN: not %run %t overflow 2>&1 | FileCheck %s

#include <omp.h>

#define M 1000
static int Data[8][M];

int main(int argc, char **argv) {
  int Overflow = argc > 1;

#pragma omp parallel num_threads(8)
  {
    int P = omp_get_thread_num();
    int N = Overflow ? M + 4096 : M;
#pragma omp target map(tofrom : Data[P : 1][0 : M]) firstprivate(P, N)
    {
      for (int I = 0; I < N; ++I)
        Data[P][I] = P + 1;
    }
  }

  for (int P = 0; P < 8; ++P)
    for (int I = 0; I < M; ++I)
      if (Data[P][I] != P + 1)
        return 1;
  return 0;
}

// CHECK: ERROR: DeviceAddressSanitizer: heap-buffer-overflow
// CHECK: WRITE of size 4
