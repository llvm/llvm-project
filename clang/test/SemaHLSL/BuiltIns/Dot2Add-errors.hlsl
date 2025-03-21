// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library %s -emit-llvm-only -disable-llvm-passes -verify

bool test_too_few_arg() {
  return __builtin_hlsl_dot2add();
  // expected-error@-1 {{too few arguments to function call, expected 3, have 0}}
}

bool test_too_many_arg(half2 p1, half2 p2, float p3) {
  return __builtin_hlsl_dot2add(p1, p2, p3, p1);
  // expected-error@-1 {{too many arguments to function call, expected 3, have 4}}
}
