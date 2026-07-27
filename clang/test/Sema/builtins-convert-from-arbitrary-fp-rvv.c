// RUN: %clang_cc1 -triple riscv64 -target-feature +v -fsyntax-only -verify %s

float sizeless_source(__rvv_uint8m1_t src) {
  return __builtin_convert_from_arbitrary_fp(src, "Float8E5M2", float); // expected-error {{first argument to __builtin_convert_from_arbitrary_fp has sizeless vector type; only fixed-length vectors are supported}}
}

__rvv_float32m1_t sizeless_destination(unsigned char src) {
  return __builtin_convert_from_arbitrary_fp(src, "Float8E5M2", __rvv_float32m1_t); // expected-error {{third argument to __builtin_convert_from_arbitrary_fp has sizeless vector type; only fixed-length vectors are supported}}
}
