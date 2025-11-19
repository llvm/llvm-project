//// REQUIRES: hexagon-registered-target || aarch64-registered-target || x86-registered-target
// RUN: %clang -g -S -O1 -emit-llvm -fenable-ripple %s -o - 2>&1 | FileCheck %s --implicit-check-not="warning:" --implicit-check-not="error:"
// RUN: %clang -g -S -O2 -emit-llvm -fenable-ripple %s -o - 2>&1 | FileCheck %s --implicit-check-not="warning:" --implicit-check-not="error:"
// RUN: %clang -g -S -O3 -emit-llvm -fenable-ripple %s -o - 2>&1 | FileCheck %s --implicit-check-not="warning:" --implicit-check-not="error:"

#include "ripple_test.h"
typedef _Bool bool;


/// Test that -fenable-ripple applies without errors
/// for certain non-SESE CFGs.
void foo(const int a[32], const int mask[32], int b[32]) {
  ripple_block_t bs = ripple_set_block_shape(0, 32);
  size_t v0 = ripple_id(bs, 0);
  bool my_cond;
  int result;
  goto external_pred;

external_pred:
  my_cond = v0 < 4;
  if (my_cond)
    goto branch_bb;
  else {
    my_cond = ~my_cond;
    goto bb_with_external_pred;
  }

branch_bb:
  my_cond = my_cond & mask[v0];
  if (my_cond) {
    result = 4;
    goto bb_with_external_pred;
  } else {
    result = 6;
    goto bb_1;
  }
bb_with_external_pred:
  result += v0 * v0 * v0;
  if (a[0])
    goto bb_2;
  else
    goto bb_1;
bb_1:
  result += v0 * v0;
  goto bb_2;
bb_2:
  b[v0] += v0*result;
  return;
}
