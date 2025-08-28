// RUN: %clang_cc1 -O1 -disable-llvm-passes -emit-llvm %s -o - 2>&1 | FileCheck %s --implicit-check-not="warning:" --implicit-check-not="error:"

#include <ripple.h>

// CHECK: @foo
int foo(int a, int b) {
  ripple_block_t BS = ripple_set_block_shape(0, 32, 4);
  // CHECK: @llvm.ripple.block.setshape.i[[REG:[0-9]+]](i[[REG]] 0, i[[REG]] 32, i[[REG]] 4, i[[REG]] 1, i[[REG]] 1, i[[REG]] 1, i[[REG]] 1, i[[REG]] 1, i[[REG]] 1, i[[REG]] 1, i[[REG]] 1)
  return a + b;
}
