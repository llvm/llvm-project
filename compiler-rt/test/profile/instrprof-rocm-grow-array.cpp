// Host unit test for __prof_rocm::growArray, the dynamic-array helper shared by
// the ROCm device-profile drains (InstrProfilingPlatformROCm.cpp and
// InstrProfilingPlatformROCmHSA.cpp). It is pure host logic with no GPU, HIP, or
// HSA dependency, so -- unlike the device drain tests under GPU/ and AMDGPU/,
// which require a real AMD GPU -- it runs anywhere the profile runtime is
// tested, including upstream CI on machines without a GPU.
//
// RUN: %clangxx %s -o %t
// RUN: %run %t | FileCheck %s

#include "../../lib/profile/InstrProfilingPlatformROCmInternal.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

using __prof_rocm::growArray;

static int Failures = 0;

#define EXPECT(Cond)                                                           \
  do {                                                                         \
    if (!(Cond)) {                                                             \
      fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, #Cond);         \
      ++Failures;                                                              \
    }                                                                          \
  } while (0)

static int allZero(const int *P, int Begin, int End) {
  for (int I = Begin; I < End; ++I)
    if (P[I] != 0)
      return 0;
  return 1;
}

int main() {
  // 1. Allocating from empty uses InitCap and zero-initializes every slot.
  {
    int *A = nullptr;
    int Cap = 0;
    EXPECT(growArray((void **)&A, &Cap, /*MinCount=*/1, /*InitCap=*/4,
                     sizeof(int)) == 0);
    EXPECT(A != nullptr);
    EXPECT(Cap == 4);
    EXPECT(allZero(A, 0, Cap));
    free(A);
  }

  // 2. Doubling continues until the capacity covers MinCount (4 -> 8 -> 16).
  {
    int *A = nullptr;
    int Cap = 0;
    EXPECT(growArray((void **)&A, &Cap, /*MinCount=*/10, /*InitCap=*/4,
                     sizeof(int)) == 0);
    EXPECT(Cap == 16);
    EXPECT(allZero(A, 0, Cap));
    free(A);
  }

  // 3. When the capacity already suffices the array is left untouched.
  {
    int *A = (int *)malloc(8 * sizeof(int));
    for (int I = 0; I < 8; ++I)
      A[I] = I + 1;
    int *Before = A;
    int Cap = 8;
    EXPECT(growArray((void **)&A, &Cap, /*MinCount=*/8, /*InitCap=*/4,
                     sizeof(int)) == 0);
    EXPECT(A == Before);
    EXPECT(Cap == 8);
    EXPECT(A[0] == 1 && A[7] == 8);
    free(A);
  }

  // 4. Growth preserves existing elements and zero-fills the new tail, with
  //    doubling resuming from the current capacity rather than InitCap.
  {
    int *A = (int *)malloc(4 * sizeof(int));
    for (int I = 0; I < 4; ++I)
      A[I] = 100 + I;
    int Cap = 4;
    EXPECT(growArray((void **)&A, &Cap, /*MinCount=*/5, /*InitCap=*/4,
                     sizeof(int)) == 0);
    EXPECT(Cap == 8);
    EXPECT(A[0] == 100 && A[1] == 101 && A[2] == 102 && A[3] == 103);
    EXPECT(allZero(A, 4, Cap));
    free(A);
  }

  // 5. ElemSize byte math is honored for wider element types.
  {
    struct Pair {
      uint64_t A, B;
    };
    Pair *P = nullptr;
    int Cap = 0;
    EXPECT(growArray((void **)&P, &Cap, /*MinCount=*/3, /*InitCap=*/2,
                     sizeof(Pair)) == 0);
    EXPECT(Cap == 4);
    int Zeroed = 1;
    for (int I = 0; I < Cap; ++I)
      if (P[I].A != 0 || P[I].B != 0)
        Zeroed = 0;
    EXPECT(Zeroed);
    free(P);
  }

  if (Failures == 0)
    printf("PASS\n");
  else
    printf("%d FAILURE(S)\n", Failures);
  return Failures != 0;
}

// CHECK: PASS
