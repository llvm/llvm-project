// REQUIRES: riscv-registered-target
// RUN: not %clang_cc1 -triple riscv64 -fsyntax-only -verify \
// RUN:   -target-feature +experimental-v %s 2>&1 \
// RUN: | FileCheck %s

#include <riscv_vector.h>

#define __rvv_generic \
static inline __attribute__((__always_inline__, __nodebug__))

__rvv_generic
__attribute__((clang_builtin_alias(__builtin_rvv_vadd_vv_i8m1)))
vint8m1_t vadd_generic (vint8m1_t op0, vint8m1_t op1, size_t op2);

// CHECK: passing 'vint8m2_t' (aka '__rvv_int8m2_t') to parameter of incompatible type 'vint8m1_t'
vint8m2_t test(vint8m2_t op0, vint8m2_t op1, size_t vl) {
  vint8m2_t ret = vadd_generic(op0, op1, vl);
  return ret;
}
