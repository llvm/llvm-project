// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library %s -emit-llvm-only -disable-llvm-passes -verify

bool test_too_few_arg() {
  return __builtin_hlsl_wave_active_all_equal();
  // expected-error@-1 {{too few arguments to function call, expected 1, have 0}}
}

bool test_too_many_arg(float2 p0) {
  return __builtin_hlsl_wave_active_all_equal(p0, p0);
  // expected-error@-1 {{too many arguments to function call, expected 1, have 2}}
}

struct S { float f; };

bool test_expr_struct_type_check(S p0) {
  return __builtin_hlsl_wave_active_all_equal(p0);
  // expected-error@-1 {{invalid operand of type 'S' where a scalar or vector is required}}
}
