// REQUIRES: aarch64-registered-target

// RUN: not %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve \
// RUN:   -target-feature +neon -S -O1 -o - %s 2>&1 | FileCheck %s

// Set a vector constraint for an sve predicate register
// As the wrong constraint is used for an SVBool,
// the compiler will try to extend the nxv16i1 to an nxv16i8
// TODO: We don't have patterns for this yet but once they are added this test
// should be updated to check for an assembler error
__SVBool_t funcB1(__SVBool_t in)
{
  __SVBool_t ret ;
  asm volatile (
    "mov %[ret].b, %[in].b \n"
    : [ret] "=w" (ret)
    : [in] "w" (in)
    :);

  return ret ;
}

// CHECK: funcB1
// CHECK-ERROR: fatal error: error in backend: Cannot select
