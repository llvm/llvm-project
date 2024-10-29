// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library %s -emit-llvm-only -disable-llvm-passes -verify

using handle_t = __hlsl_resource_t [[hlsl::resource_class(UAV)]] [[hlsl::contained_type(int)]];

void test_args(int x, bool b) {
  handle_t res;

  // expected-error@+1 {{too few arguments to function call, expected 2, have 1}}
  __builtin_hlsl_buffer_update_counter(x);

  // expected-error@+1 {{too many arguments to function call, expected 2, have 3}}
  __builtin_hlsl_buffer_update_counter(x, x, x);

  // expected-error@+1 {{used type 'int' where __hlsl_resource_t is required}}
  __builtin_hlsl_buffer_update_counter(x, x);

  // expected-error@+1 {{argument 1 must be constant integer 1 or -1}}
  __builtin_hlsl_buffer_update_counter(res, x);

  // expected-error@+1 {{argument 1 must be constant integer 1 or -1}}
  __builtin_hlsl_buffer_update_counter(res, 10);
}
