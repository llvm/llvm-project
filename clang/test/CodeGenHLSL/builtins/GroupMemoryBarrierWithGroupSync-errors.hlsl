// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library %s -fnative-half-type -emit-llvm-only -disable-llvm-passes -verify -verify-ignore-unexpected

void test_too_many_arg() {
  __builtin_group_memory_barrier_with_group_sync(0);
  // expected-error@-1 {{too many arguments to function call, expected at most 0, have 1}}
}
