// RUN: %clang_cc1 -triple riscv64 -target-feature +v -fsyntax-only -verify %s

__rvv_float32m1_t convert(__rvv_uint8m1_t src) {
  return __builtin_elementwise_convert_from_f8e5m2_f32(src); // expected-error {{1st argument must be a scalar or vector of 8-bit integer types (was '__rvv_uint8m1_t')}}
}
