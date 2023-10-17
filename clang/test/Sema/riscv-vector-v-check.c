// RUN: %clang_cc1 -triple riscv64 -target-feature +zve64x \
// RUN:   -disable-O0-optnone -o - -fsyntax-only %s -verify 
// REQUIRES: riscv-registered-target
#include <riscv_vector.h>

vint64m1_t test_vsmul_vv_i64m1(vint64m1_t op1, vint64m1_t op2, size_t vl) {
  return __riscv_vsmul_vv_i64m1(op1, op2, __RISCV_VXRM_RNU, vl); /* expected-error {{builtin requires: v}} */
}
vint64m1_t test_vsmul_vx_i64m1(vint64m1_t op1, int64_t op2, size_t vl) {
  return __riscv_vsmul_vx_i64m1(op1, op2, __RISCV_VXRM_RNU, vl); /* expected-error {{builtin requires: v}} */
}
vint64m2_t test_vsmul_vv_i64m2(vint64m2_t op1, vint64m2_t op2, size_t vl) {
  return __riscv_vsmul_vv_i64m2(op1, op2, __RISCV_VXRM_RNU, vl); /* expected-error {{builtin requires: v}} */
}
vint64m2_t test_vsmul_vx_i64m2(vint64m2_t op1, int64_t op2, size_t vl) {
  return __riscv_vsmul_vx_i64m2(op1, op2, __RISCV_VXRM_RNU, vl); /* expected-error {{builtin requires: v}} */
}
vint64m4_t test_vsmul_vv_i64m4(vint64m4_t op1, vint64m4_t op2, size_t vl) {
  return __riscv_vsmul_vv_i64m4(op1, op2, __RISCV_VXRM_RNU, vl); /* expected-error {{builtin requires: v}} */
}
vint64m4_t test_vsmul_vx_i64m4(vint64m4_t op1, int64_t op2, size_t vl) {
  return __riscv_vsmul_vx_i64m4(op1, op2, __RISCV_VXRM_RNU, vl); /* expected-error {{builtin requires: v}} */
}
vint64m8_t test_vsmul_vv_i64m8(vint64m8_t op1, vint64m8_t op2, size_t vl) {
  return __riscv_vsmul_vv_i64m8(op1, op2, __RISCV_VXRM_RNU, vl); /* expected-error {{builtin requires: v}} */
}
vint64m8_t test_vsmul_vx_i64m8(vint64m8_t op1, int64_t op2, size_t vl) {
  return __riscv_vsmul_vx_i64m8(op1, op2, __RISCV_VXRM_RNU, vl); /* expected-error {{builtin requires: v}} */
}
vint64m1_t test_vsmul_vv_i64m1_m(vbool64_t mask, vint64m1_t op1, vint64m1_t op2, size_t vl) {
  return __riscv_vsmul_vv_i64m1_m(mask, op1, op2, __RISCV_VXRM_RNU, vl); /* expected-error {{builtin requires: v}} */
}
vint64m1_t test_vsmul_vx_i64m1_m(vbool64_t mask, vint64m1_t op1, int64_t op2, size_t vl) {
  return __riscv_vsmul_vx_i64m1_m(mask, op1, op2, __RISCV_VXRM_RNU, vl); /* expected-error {{builtin requires: v}} */
}
vint64m2_t test_vsmul_vv_i64m2_m(vbool32_t mask, vint64m2_t op1, vint64m2_t op2, size_t vl) {
  return __riscv_vsmul_vv_i64m2_m(mask, op1, op2, __RISCV_VXRM_RNU, vl); /* expected-error {{builtin requires: v}} */
}
vint64m2_t test_vsmul_vx_i64m2_m(vbool32_t mask, vint64m2_t op1, int64_t op2, size_t vl) {
  return __riscv_vsmul_vx_i64m2_m(mask, op1, op2, __RISCV_VXRM_RNU, vl); /* expected-error {{builtin requires: v}} */
}
vint64m4_t test_vsmul_vv_i64m4_m(vbool16_t mask, vint64m4_t op1, vint64m4_t op2, size_t vl) {
  return __riscv_vsmul_vv_i64m4_m(mask, op1, op2, __RISCV_VXRM_RNU, vl); /* expected-error {{builtin requires: v}} */
}
vint64m4_t test_vsmul_vx_i64m4_m(vbool16_t mask, vint64m4_t op1, int64_t op2, size_t vl) {
  return __riscv_vsmul_vx_i64m4_m(mask, op1, op2, __RISCV_VXRM_RNU, vl); /* expected-error {{builtin requires: v}} */
}
vint64m8_t test_vsmul_vv_i64m8_m(vbool8_t mask, vint64m8_t op1, vint64m8_t op2, size_t vl) {
  return __riscv_vsmul_vv_i64m8_m(mask, op1, op2, __RISCV_VXRM_RNU, vl); /* expected-error {{builtin requires: v}} */
}
vint64m8_t test_vsmul_vx_i64m8_m(vbool8_t mask, vint64m8_t op1, int64_t op2, size_t vl) {
  return __riscv_vsmul_vx_i64m8_m(mask, op1, op2, __RISCV_VXRM_RNU, vl); /* expected-error {{builtin requires: v}} */
}
vint64m1_t test_vmulh_vv_i64m1(vint64m1_t op1, vint64m1_t op2, size_t vl) {
  return __riscv_vmulh_vv_i64m1(op1, op2, vl); /* expected-error {{builtin requires: v}} */
}
vint64m1_t test_vmulh_vx_i64m1(vint64m1_t op1, int64_t op2, size_t vl) {
  return __riscv_vmulh_vx_i64m1(op1, op2, vl); /* expected-error {{builtin requires: v}} */
}
vint64m2_t test_vmulh_vv_i64m2(vint64m2_t op1, vint64m2_t op2, size_t vl) {
  return __riscv_vmulh_vv_i64m2(op1, op2, vl); /* expected-error {{builtin requires: v}} */
}
vint64m2_t test_vmulh_vx_i64m2(vint64m2_t op1, int64_t op2, size_t vl) {
  return __riscv_vmulh_vx_i64m2(op1, op2, vl); /* expected-error {{builtin requires: v}} */
}
vint64m4_t test_vmulh_vv_i64m4(vint64m4_t op1, vint64m4_t op2, size_t vl) {
  return __riscv_vmulh_vv_i64m4(op1, op2, vl); /* expected-error {{builtin requires: v}} */
}
vint64m4_t test_vmulh_vx_i64m4(vint64m4_t op1, int64_t op2, size_t vl) {
  return __riscv_vmulh_vx_i64m4(op1, op2, vl); /* expected-error {{builtin requires: v}} */
}
vint64m8_t test_vmulh_vv_i64m8(vint64m8_t op1, vint64m8_t op2, size_t vl) {
  return __riscv_vmulh_vv_i64m8(op1, op2, vl); /* expected-error {{builtin requires: v}} */
}
vint64m8_t test_vmulh_vx_i64m8(vint64m8_t op1, int64_t op2, size_t vl) {
  return __riscv_vmulh_vx_i64m8(op1, op2, vl); /* expected-error {{builtin requires: v}} */
}
vint64m1_t test_vmulh_vv_i64m1_m(vbool64_t mask, vint64m1_t op1, vint64m1_t op2, size_t vl) {
  return __riscv_vmulh_vv_i64m1_m(mask, op1, op2, vl); /* expected-error {{builtin requires: v}} */
}
vint64m1_t test_vmulh_vx_i64m1_m(vbool64_t mask, vint64m1_t op1, int64_t op2, size_t vl) {
  return __riscv_vmulh_vx_i64m1_m(mask, op1, op2, vl); /* expected-error {{builtin requires: v}} */
}
vint64m2_t test_vmulh_vv_i64m2_m(vbool32_t mask, vint64m2_t op1, vint64m2_t op2, size_t vl) {
  return __riscv_vmulh_vv_i64m2_m(mask, op1, op2, vl); /* expected-error {{builtin requires: v}} */
}
vint64m2_t test_vmulh_vx_i64m2_m(vbool32_t mask, vint64m2_t op1, int64_t op2, size_t vl) {
  return __riscv_vmulh_vx_i64m2_m(mask, op1, op2, vl); /* expected-error {{builtin requires: v}} */
}
vint64m4_t test_vmulh_vv_i64m4_m(vbool16_t mask, vint64m4_t op1, vint64m4_t op2, size_t vl) {
  return __riscv_vmulh_vv_i64m4_m(mask, op1, op2, vl); /* expected-error {{builtin requires: v}} */
}
vint64m4_t test_vmulh_vx_i64m4_m(vbool16_t mask, vint64m4_t op1, int64_t op2, size_t vl) {
  return __riscv_vmulh_vx_i64m4_m(mask, op1, op2, vl); /* expected-error {{builtin requires: v}} */
}
vint64m8_t test_vmulh_vv_i64m8_m(vbool8_t mask, vint64m8_t op1, vint64m8_t op2, size_t vl) {
  return __riscv_vmulh_vv_i64m8_m(mask, op1, op2, vl); /* expected-error {{builtin requires: v}} */
}
vint64m8_t test_vmulh_vx_i64m8_m(vbool8_t mask, vint64m8_t op1, int64_t op2, size_t vl) {
  return __riscv_vmulh_vx_i64m8_m(mask, op1, op2, vl); /* expected-error {{builtin requires: v}} */
}
vuint64m1_t test_vmulhu_vv_u64m1(vuint64m1_t op1, vuint64m1_t op2, size_t vl) {
  return __riscv_vmulhu_vv_u64m1(op1, op2, vl); /* expected-error {{builtin requires: v}} */
}
vuint64m1_t test_vmulhu_vx_u64m1(vuint64m1_t op1, uint64_t op2, size_t vl) {
  return __riscv_vmulhu_vx_u64m1(op1, op2, vl); /* expected-error {{builtin requires: v}} */
}
vuint64m2_t test_vmulhu_vv_u64m2(vuint64m2_t op1, vuint64m2_t op2, size_t vl) {
  return __riscv_vmulhu_vv_u64m2(op1, op2, vl); /* expected-error {{builtin requires: v}} */
}
vuint64m2_t test_vmulhu_vx_u64m2(vuint64m2_t op1, uint64_t op2, size_t vl) {
  return __riscv_vmulhu_vx_u64m2(op1, op2, vl); /* expected-error {{builtin requires: v}} */
}
vuint64m4_t test_vmulhu_vv_u64m4(vuint64m4_t op1, vuint64m4_t op2, size_t vl) {
  return __riscv_vmulhu_vv_u64m4(op1, op2, vl); /* expected-error {{builtin requires: v}} */
}
vuint64m4_t test_vmulhu_vx_u64m4(vuint64m4_t op1, uint64_t op2, size_t vl) {
  return __riscv_vmulhu_vx_u64m4(op1, op2, vl); /* expected-error {{builtin requires: v}} */
}
vuint64m8_t test_vmulhu_vv_u64m8(vuint64m8_t op1, vuint64m8_t op2, size_t vl) {
  return __riscv_vmulhu_vv_u64m8(op1, op2, vl); /* expected-error {{builtin requires: v}} */
}
vuint64m8_t test_vmulhu_vx_u64m8(vuint64m8_t op1, uint64_t op2, size_t vl) {
  return __riscv_vmulhu_vx_u64m8(op1, op2, vl); /* expected-error {{builtin requires: v}} */
}
vuint64m1_t test_vmulhu_vv_u64m1_m(vbool64_t mask, vuint64m1_t op1, vuint64m1_t op2, size_t vl) {
  return __riscv_vmulhu_vv_u64m1_m(mask, op1, op2, vl); /* expected-error {{builtin requires: v}} */
}
vuint64m1_t test_vmulhu_vx_u64m1_m(vbool64_t mask, vuint64m1_t op1, uint64_t op2, size_t vl) {
  return __riscv_vmulhu_vx_u64m1_m(mask, op1, op2, vl); /* expected-error {{builtin requires: v}} */
}
vuint64m2_t test_vmulhu_vv_u64m2_m(vbool32_t mask, vuint64m2_t op1, vuint64m2_t op2, size_t vl) {
  return __riscv_vmulhu_vv_u64m2_m(mask, op1, op2, vl); /* expected-error {{builtin requires: v}} */
}
vuint64m2_t test_vmulhu_vx_u64m2_m(vbool32_t mask, vuint64m2_t op1, uint64_t op2, size_t vl) {
  return __riscv_vmulhu_vx_u64m2_m(mask, op1, op2, vl); /* expected-error {{builtin requires: v}} */
}
vuint64m4_t test_vmulhu_vv_u64m4_m(vbool16_t mask, vuint64m4_t op1, vuint64m4_t op2, size_t vl) {
  return __riscv_vmulhu_vv_u64m4_m(mask, op1, op2, vl); /* expected-error {{builtin requires: v}} */
}
vuint64m4_t test_vmulhu_vx_u64m4_m(vbool16_t mask, vuint64m4_t op1, uint64_t op2, size_t vl) {
  return __riscv_vmulhu_vx_u64m4_m(mask, op1, op2, vl); /* expected-error {{builtin requires: v}} */
}
vuint64m8_t test_vmulhu_vv_u64m8_m(vbool8_t mask, vuint64m8_t op1, vuint64m8_t op2, size_t vl) {
  return __riscv_vmulhu_vv_u64m8_m(mask, op1, op2, vl); /* expected-error {{builtin requires: v}} */
}
vuint64m8_t test_vmulhu_vx_u64m8_m(vbool8_t mask, vuint64m8_t op1, uint64_t op2, size_t vl) {
  return __riscv_vmulhu_vx_u64m8_m(mask, op1, op2, vl); /* expected-error {{builtin requires: v}} */
}
vint64m1_t test_vmulhsu_vv_i64m1(vint64m1_t op1, vuint64m1_t op2, size_t vl) {
  return __riscv_vmulhsu_vv_i64m1(op1, op2, vl); /* expected-error {{builtin requires: v}} */
}
vint64m1_t test_vmulhsu_vx_i64m1(vint64m1_t op1, uint64_t op2, size_t vl) {
  return __riscv_vmulhsu_vx_i64m1(op1, op2, vl); /* expected-error {{builtin requires: v}} */
}
vint64m2_t test_vmulhsu_vv_i64m2(vint64m2_t op1, vuint64m2_t op2, size_t vl) {
  return __riscv_vmulhsu_vv_i64m2(op1, op2, vl); /* expected-error {{builtin requires: v}} */
}
vint64m2_t test_vmulhsu_vx_i64m2(vint64m2_t op1, uint64_t op2, size_t vl) {
  return __riscv_vmulhsu_vx_i64m2(op1, op2, vl); /* expected-error {{builtin requires: v}} */
}
vint64m4_t test_vmulhsu_vv_i64m4(vint64m4_t op1, vuint64m4_t op2, size_t vl) {
  return __riscv_vmulhsu_vv_i64m4(op1, op2, vl); /* expected-error {{builtin requires: v}} */
}
vint64m4_t test_vmulhsu_vx_i64m4(vint64m4_t op1, uint64_t op2, size_t vl) {
  return __riscv_vmulhsu_vx_i64m4(op1, op2, vl); /* expected-error {{builtin requires: v}} */
}
vint64m8_t test_vmulhsu_vv_i64m8(vint64m8_t op1, vuint64m8_t op2, size_t vl) {
  return __riscv_vmulhsu_vv_i64m8(op1, op2, vl); /* expected-error {{builtin requires: v}} */
}
vint64m8_t test_vmulhsu_vx_i64m8(vint64m8_t op1, uint64_t op2, size_t vl) {
  return __riscv_vmulhsu_vx_i64m8(op1, op2, vl); /* expected-error {{builtin requires: v}} */
}
vint64m1_t test_vmulhsu_vv_i64m1_m(vbool64_t mask, vint64m1_t op1, vuint64m1_t op2, size_t vl) {
  return __riscv_vmulhsu_vv_i64m1_m(mask, op1, op2, vl); /* expected-error {{builtin requires: v}} */
}
vint64m1_t test_vmulhsu_vx_i64m1_m(vbool64_t mask, vint64m1_t op1, uint64_t op2, size_t vl) {
  return __riscv_vmulhsu_vx_i64m1_m(mask, op1, op2, vl); /* expected-error {{builtin requires: v}} */
}
vint64m2_t test_vmulhsu_vv_i64m2_m(vbool32_t mask, vint64m2_t op1, vuint64m2_t op2, size_t vl) {
  return __riscv_vmulhsu_vv_i64m2_m(mask, op1, op2, vl); /* expected-error {{builtin requires: v}} */
}
vint64m2_t test_vmulhsu_vx_i64m2_m(vbool32_t mask, vint64m2_t op1, uint64_t op2, size_t vl) {
  return __riscv_vmulhsu_vx_i64m2_m(mask, op1, op2, vl); /* expected-error {{builtin requires: v}} */
}
vint64m4_t test_vmulhsu_vv_i64m4_m(vbool16_t mask, vint64m4_t op1, vuint64m4_t op2, size_t vl) {
  return __riscv_vmulhsu_vv_i64m4_m(mask, op1, op2, vl); /* expected-error {{builtin requires: v}} */
}
vint64m4_t test_vmulhsu_vx_i64m4_m(vbool16_t mask, vint64m4_t op1, uint64_t op2, size_t vl) {
  return __riscv_vmulhsu_vx_i64m4_m(mask, op1, op2, vl); /* expected-error {{builtin requires: v}} */
}
vint64m8_t test_vmulhsu_vv_i64m8_m(vbool8_t mask, vint64m8_t op1, vuint64m8_t op2, size_t vl) {
  return __riscv_vmulhsu_vv_i64m8_m(mask, op1, op2, vl); /* expected-error {{builtin requires: v}} */
}
vint64m8_t test_vmulhsu_vx_i64m8_m(vbool8_t mask, vint64m8_t op1, uint64_t op2, size_t vl) {
  return __riscv_vmulhsu_vx_i64m8_m(mask, op1, op2, vl); /* expected-error {{builtin requires: v}} */
}
