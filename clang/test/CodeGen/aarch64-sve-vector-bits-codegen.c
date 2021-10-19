// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -O2 -S -o - %s -msve-vector-bits=256  | FileCheck %s --check-prefixes=CHECK,CHECK256
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -O2 -S -o - %s -msve-vector-bits=512  | FileCheck %s --check-prefixes=CHECK,CHECK512
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -O2 -S -o - %s -msve-vector-bits=1024 | FileCheck %s --check-prefixes=CHECK,CHECK1024
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -O2 -S -o - %s -msve-vector-bits=2048 | FileCheck %s --check-prefixes=CHECK,CHECK2048
// REQUIRES: aarch64-registered-target

#include <arm_sve.h>

void func(int *restrict a, int *restrict b) {
// CHECK-LABEL: func
// CHECK256-COUNT-8: st1w
// CHECK512-COUNT-4: st1w
// CHECK1024-COUNT-2: st1w
// CHECK2048-COUNT-1: st1w
#pragma clang loop vectorize(enable)
  for (int i = 0; i < 64; ++i)
    a[i] += b[i];
}
