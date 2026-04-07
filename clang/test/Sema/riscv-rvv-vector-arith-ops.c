// RUN: %clang_cc1 -triple riscv32 -target-feature +v -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple riscv64 -target-feature +v -fsyntax-only -verify %s

// REQUIRES: riscv-registered-target

#include <riscv_vector.h>

// Arithmetic on mask types is not allowed
vbool32_t bad_add_bool(vbool32_t a, vbool32_t b) {
  return a + b; // expected-error {{invalid operands to binary expression}}
}

// Bitwise NOT on floats is not allowed
vfloat32m1_t bad_not_f32(vfloat32m1_t a) {
  return ~a; // expected-error {{invalid argument type}}
}

// Remainder on floats is not allowed
vfloat32m1_t bad_rem_f32(vfloat32m1_t a, vfloat32m1_t b) {
  return a % b; // expected-error {{invalid operands to binary expression}}
}

// Shift on floats is not allowed
vfloat32m1_t bad_shl_f32(vfloat32m1_t a, vfloat32m1_t b) {
  return a << b; // expected-error {{used type 'vfloat32m1_t'}}
}

// Bitwise ops on floats are not allowed
vfloat32m1_t bad_and_f32(vfloat32m1_t a, vfloat32m1_t b) {
  return a & b; // expected-error {{invalid operands to binary expression}}
}

// Mismatched vector types
vint32m1_t bad_add_mismatch(vint32m1_t a, vint64m1_t b) {
  return a + b; // expected-error {{vector operands do not have the same number of elements}}
}

// Mismatched LMUL
vint32m1_t bad_add_lmul(vint32m1_t a, vint32m2_t b) {
  return a + b; // expected-error {{vector operands do not have the same number of elements}}
}

// Subscript on bool is not allowed
int bad_subscript_bool(vbool32_t a) {
  return a[0]; // expected-error {{subscript of svbool_t is not allowed}}
}

// Subscript with float index is not allowed
int bad_subscript_float_idx(vint32m1_t a) {
  return a[0.f]; // expected-error {{array subscript is not an integer}}
}

int bad_subscript_double_idx(vint32m1_t a) {
  return a[0.]; // expected-error {{array subscript is not an integer}}
}

// Compound assignment on bools is not allowed
void bad_compound_bool(vbool32_t a, vbool32_t b) {
  a += b; // expected-error {{invalid operands to binary expression}}
}

// Mixed signedness (same element size and LMUL) is not allowed
vint32m1_t bad_add_mixed_sign(vint32m1_t a, vuint32m1_t b) {
  return a + b; // expected-error {{cannot convert between vector type 'vuint32m1_t'}}
}

// Mixed element width (different element count) is not allowed
vint32m1_t bad_add_mixed_width(vint32m1_t a, vuint64m1_t b) {
  return a + b; // expected-error {{vector operands do not have the same number of elements}}
}

// Same element count but different element width (i32m1 + i64m2 are both
// vscale x 2, but i32 vs i64) - this is an RVV-specific case that cannot
// occur with SVE where element count always differs with element width.
vint64m2_t bad_add_same_ec_diff_eltty(vint32m1_t a, vint64m2_t b) {
  return a + b; // expected-error {{cannot convert between vector type}}
}
