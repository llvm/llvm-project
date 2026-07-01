// Host unit test for __prof_rocm::ProfBoundsSet, the section-bounds dedup table
// shared by the ROCm device-profile drains (InstrProfilingPlatformROCm.cpp and
// InstrProfilingPlatformROCmHSA.cpp). This is the bookkeeping that guarantees a
// device counter set is drained exactly once -- across the host-shadow and HSA
// paths and across the multiple GPU agents that may share a code object (the
// "device bounds already drained, skipping" behavior exercised by the multi-GPU
// device test). It is pure host logic with no GPU/HIP/HSA dependency, so unlike
// the device drain tests under GPU/ and AMDGPU/ it runs anywhere the profile
// runtime is tested, including upstream CI on machines without an AMD GPU.
//
// RUN: %clangxx %s -o %t
// RUN: %run %t | FileCheck %s

#include "../../lib/profile/InstrProfilingPlatformROCmInternal.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

using __prof_rocm::ProfBoundsSet;

static int Failures = 0;

#define EXPECT(Cond)                                                           \
  do {                                                                         \
    if (!(Cond)) {                                                             \
      fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, #Cond);         \
      ++Failures;                                                              \
    }                                                                          \
  } while (0)

// Distinct, non-null fake section pointers derived from an integer.
static const void *P(uintptr_t V) { return (const void *)(V * 8 + 8); }

int main() {
  // 1. A fresh set contains nothing.
  {
    ProfBoundsSet S;
    EXPECT(S.Count == 0);
    EXPECT(!S.contains(P(1), P(2), P(3)));
    free(S.Items);
  }

  // 2. record() is idempotent: the first insert reports "new", repeats do not,
  //    and the element count never double-counts.
  {
    ProfBoundsSet S;
    EXPECT(S.record(P(1), P(2), P(3)) == true);
    EXPECT(S.contains(P(1), P(2), P(3)));
    EXPECT(S.Count == 1);
    EXPECT(S.record(P(1), P(2), P(3)) == false);
    EXPECT(S.record(P(1), P(2), P(3)) == false);
    EXPECT(S.Count == 1);
    free(S.Items);
  }

  // 3. All three fields are part of the key: differing in any single field
  //    (data, counters, or names) is a distinct tuple. Guards against a dedup
  //    that keys on only a subset and would drop a real counter set.
  {
    ProfBoundsSet S;
    EXPECT(S.record(P(1), P(2), P(3)) == true);
    EXPECT(!S.contains(P(9), P(2), P(3))); // data differs
    EXPECT(!S.contains(P(1), P(9), P(3))); // counters differ
    EXPECT(!S.contains(P(1), P(2), P(9))); // names differ
    EXPECT(S.record(P(9), P(2), P(3)) == true);
    EXPECT(S.record(P(1), P(9), P(3)) == true);
    EXPECT(S.record(P(1), P(2), P(9)) == true);
    EXPECT(S.Count == 4);
    free(S.Items);
  }

  // 4. Many distinct tuples grow the table past its initial capacity; all stay
  //    recorded and re-recording any of them is still a no-op.
  {
    ProfBoundsSet S;
    const int N = 4 * ProfBoundsSet::kInitCap + 7; // forces several doublings
    for (int I = 0; I < N; ++I)
      EXPECT(S.record(P(3 * I + 1), P(3 * I + 2), P(3 * I + 3)) == true);
    EXPECT(S.Count == N);
    EXPECT(S.Cap >= N);
    for (int I = 0; I < N; ++I) {
      EXPECT(S.contains(P(3 * I + 1), P(3 * I + 2), P(3 * I + 3)));
      EXPECT(S.record(P(3 * I + 1), P(3 * I + 2), P(3 * I + 3)) == false);
    }
    EXPECT(S.Count == N); // duplicates did not grow the table
    free(S.Items);
  }

  // 5. Null pointers are valid keys (an empty/zero code object is recorded so a
  //    later agent skips it rather than reprocessing it).
  {
    ProfBoundsSet S;
    EXPECT(S.record(nullptr, nullptr, nullptr) == true);
    EXPECT(S.contains(nullptr, nullptr, nullptr));
    EXPECT(S.record(nullptr, nullptr, nullptr) == false);
    EXPECT(S.Count == 1);
    free(S.Items);
  }

  if (Failures == 0)
    printf("PASS\n");
  else
    printf("%d FAILURE(S)\n", Failures);
  return Failures != 0;
}

// CHECK: PASS
