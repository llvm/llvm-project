// REQUIRES: aarch64-registered-target

// RUN: not %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve \
// RUN:   -target-feature +neon -S -O1 -o - %s | FileCheck %s

// Assembler error
// Output constraint : Set a vector constraint on an integer
__SVFloat32_t funcB2()
{
  __SVFloat32_t ret ;
  asm volatile (
    "fmov %[ret], wzr \n"
    : [ret] "=w" (ret)
    :
    :);

  return ret ;
}

// CHECK: funcB2
// CHECK-ERROR: error: invalid operand for instruction
