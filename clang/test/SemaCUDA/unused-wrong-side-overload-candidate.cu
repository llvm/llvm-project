// RUN: %clang_cc1 -std=c++20 -x hip -fcuda-is-device \
// RUN:   -foffload-implicit-host-device-templates -Wall -Werror \
// RUN:   -fsyntax-only %s
// RUN: %clang_cc1 -std=c++20 -x hip \
// RUN:   -foffload-implicit-host-device-templates -Wall -Werror \
// RUN:   -fsyntax-only %s

#include "Inputs/cuda.h"

struct BothTy {};

static bool operator==(BothTy, BothTy) { return true; }
static __device__ bool operator==(BothTy, BothTy) { return true; }

template <class T> bool compare(T LHS, T RHS) {
  return LHS == RHS;
}

__host__ bool host_use() {
  return compare(BothTy{}, BothTy{});
}

__device__ bool device_use() {
  return compare(BothTy{}, BothTy{});
}
