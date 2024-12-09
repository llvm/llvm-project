// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library %s -emit-llvm-only -disable-llvm-passes -verify

// RWBuffer<int>
using handle_t = __hlsl_resource_t
    [[hlsl::resource_class(UAV)]] [[hlsl::contained_type(int)]];

void test_args(unsigned int x) {
  // expected-error@+1 {{too few arguments to function call, expected 2, have 1}}
  __builtin_hlsl_resource_getpointer(x);

  // expected-error@+1 {{too many arguments to function call, expected 2, have 3}}
  __builtin_hlsl_resource_getpointer(x, x, x);

  // expected-error@+1 {{used type 'unsigned int' where __hlsl_resource_t is required}}
  __builtin_hlsl_resource_getpointer(x, x);

  handle_t res;

  // expected-error@+1 {{passing 'const char *' to parameter of incompatible type 'unsigned int'}}
  __builtin_hlsl_resource_getpointer(res, "1");

  // no error
  __builtin_hlsl_resource_getpointer(res, 0u);

  // no error
  __builtin_hlsl_resource_getpointer(res, x);
}
