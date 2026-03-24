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

  // expected-error@+1 {{used type 'const char *' where integer is required}}
  __builtin_hlsl_resource_getpointer(res, "1");

  // no error
  __builtin_hlsl_resource_getpointer(res, 0u);

  // no error
  __builtin_hlsl_resource_getpointer(res, x);

  // expected-error@+1 {{builtin '__builtin_hlsl_resource_getpointer' resource coordinate dimension mismatch: expected 1, found 2}}
  __builtin_hlsl_resource_getpointer(res, uint2(1, 2));
}

using tex2d_handle_t = __hlsl_resource_t
    [[hlsl::resource_class(SRV)]] [[hlsl::dimension("2D")]] [[hlsl::contained_type(float4)]];

using tex3d_handle_t = __hlsl_resource_t
    [[hlsl::resource_class(SRV)]] [[hlsl::dimension("3D")]] [[hlsl::contained_type(float4)]];

void test_tex_handles(tex2d_handle_t tex2d, tex3d_handle_t tex3d) {
  // expected-error@+1 {{builtin '__builtin_hlsl_resource_getpointer' resource coordinate dimension mismatch: expected 2, found 1}}
  __builtin_hlsl_resource_getpointer(tex2d, 1u);

  // no error
  __builtin_hlsl_resource_getpointer(tex2d, uint2(1, 2));

  // expected-error@+1 {{builtin '__builtin_hlsl_resource_getpointer' resource coordinate dimension mismatch: expected 2, found 3}}
  __builtin_hlsl_resource_getpointer(tex2d, uint3(1, 2, 3));

  // expected-error@+1 {{builtin '__builtin_hlsl_resource_getpointer' resource coordinate dimension mismatch: expected 3, found 2}}
  __builtin_hlsl_resource_getpointer(tex3d, uint2(1, 2));

  // no error
  __builtin_hlsl_resource_getpointer(tex3d, uint3(1, 2, 3));
}
