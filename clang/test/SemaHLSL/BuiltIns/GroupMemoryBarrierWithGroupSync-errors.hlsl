// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library %s -emit-llvm-only -disable-llvm-passes -verify

void test_too_many_arg() {
  __builtin_hlsl_group_memory_barrier_with_group_sync(0);
  // expected-error@-1 {{too many arguments to function call, expected 0, have 1}}
}
