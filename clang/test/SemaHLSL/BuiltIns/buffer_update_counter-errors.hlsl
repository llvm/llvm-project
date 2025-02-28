// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library %s -emit-llvm-only -disable-llvm-passes -verify

// RWStructuredBuffer<int>
using handle_t = __hlsl_resource_t [[hlsl::resource_class(UAV)]] [[hlsl::contained_type(int)]] [[hlsl::raw_buffer]];
// RWBuffer<int>
using bad_handle_not_raw_t = __hlsl_resource_t [[hlsl::resource_class(UAV)]] [[hlsl::contained_type(int)]];
// RWByteAddressBuffer
using bad_handle_no_type_t = __hlsl_resource_t [[hlsl::resource_class(UAV)]] [[hlsl::raw_buffer]];
// StructuredBuffer
using bad_handle_not_uav_t = __hlsl_resource_t [[hlsl::resource_class(SRV)]] [[hlsl::contained_type(int)]] [[hlsl::raw_buffer]];

void test_args(int x, bool b) {
  // expected-error@+1 {{too few arguments to function call, expected 2, have 1}}
  __builtin_hlsl_buffer_update_counter(x);

  // expected-error@+1 {{too many arguments to function call, expected 2, have 3}}
  __builtin_hlsl_buffer_update_counter(x, x, x);

  // expected-error@+1 {{used type 'int' where __hlsl_resource_t is required}}
  __builtin_hlsl_buffer_update_counter(x, x);

  bad_handle_not_raw_t bad1;
  bad_handle_no_type_t bad2;
  bad_handle_not_uav_t bad3;

  // expected-error@+1 {{invalid __hlsl_resource_t type attributes}}
  __builtin_hlsl_buffer_update_counter(bad1, 1);

  // expected-error@+1 {{invalid __hlsl_resource_t type attributes}}
  __builtin_hlsl_buffer_update_counter(bad2, 1);

  // expected-error@+1 {{invalid __hlsl_resource_t type attributes}}
  __builtin_hlsl_buffer_update_counter(bad3, 1);

  handle_t res;

  // expected-error@+1 {{argument 1 must be constant integer 1 or -1}}
  __builtin_hlsl_buffer_update_counter(res, x);

  // expected-error@+1 {{passing 'const char *' to parameter of incompatible type 'int'}}
  __builtin_hlsl_buffer_update_counter(res, "1");
  
  // expected-error@+1 {{argument 1 must be constant integer 1 or -1}}
  __builtin_hlsl_buffer_update_counter(res, 10);

  // no error
  __builtin_hlsl_buffer_update_counter(res, 1);
}
