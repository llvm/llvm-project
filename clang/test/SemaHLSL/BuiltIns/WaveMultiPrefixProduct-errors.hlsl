// RUN: %clang_cc1 -finclude-default-header \
// RUN:   -fnative-half-type -fnative-int16-type \
// RUN:   -triple dxil-pc-shadermodel6.5-library -verify %s
// RUN: %clang_cc1 -finclude-default-header \
// RUN:   -fnative-half-type -fnative-int16-type \
// RUN:   -triple spirv-pc-vulkan-library -verify %s
// RUN: %clang_cc1 -finclude-default-header -DTEST_AVAILABILITY \
// RUN:   -triple dxil-pc-shadermodel6.4-library -verify=availability %s

#ifdef TEST_AVAILABILITY
[shader("compute")]
[numthreads(1, 1, 1)]
void main() {
  (void)WaveMultiPrefixProduct(1.0f, uint4(1, 2, 4, 8));
  // availability-error@-1 {{only available on Shader Model 6.5 or newer}}
  // availability-note@*:* {{introduced in Shader Model 6.5}}
}
#else
void test_arg_count(float value, uint4 mask) {
  __builtin_hlsl_wave_multi_prefix_product(value);
  // expected-error@-1 {{too few arguments to function call, expected 2}}
  __builtin_hlsl_wave_multi_prefix_product(value, mask, mask);
  // expected-error@-1 {{too many arguments to function call, expected 2}}
}

struct S { float value; };

void test_value_types(bool b, bool2 bv, bool2x2 bm, S s, uint4 mask) {
  __builtin_hlsl_wave_multi_prefix_product(b, mask);
  // expected-error@-1 {{invalid operand of type 'bool'}}
  __builtin_hlsl_wave_multi_prefix_product(bv, mask);
  // expected-error@-1 {{invalid operand of type 'bool2'}}
  __builtin_hlsl_wave_multi_prefix_product(bm, mask);
  // expected-error@-1 {{1st argument must be a matrix, integer or \
floating-point type (was 'bool2x2' (aka 'matrix<bool, 2, 2>'))}}
  // expected-error@-3 {{cannot pass object of non-trivial type 'bool2x2'}}
  __builtin_hlsl_wave_multi_prefix_product(s, mask);
  // expected-error@-1 {{invalid operand of type 'S'}}
}

void test_mask_types(float value, uint scalar, uint3 short_mask,
                     int4 signed_mask, float4 float_mask,
                     uint64_t4 wide_mask, S struct_mask) {
  __builtin_hlsl_wave_multi_prefix_product(value, scalar);
  // expected-error@-1 {{passing 'uint' (aka 'unsigned int')}}
  __builtin_hlsl_wave_multi_prefix_product(value, short_mask);
  // expected-error@-1 {{passing 'uint3' (aka 'vector<uint, 3>')}}
  __builtin_hlsl_wave_multi_prefix_product(value, signed_mask);
  // expected-error@-1 {{passing 'int4' (aka 'vector<int, 4>')}}
  __builtin_hlsl_wave_multi_prefix_product(value, float_mask);
  // expected-error@-1 {{passing 'float4' (aka 'vector<float, 4>')}}
  __builtin_hlsl_wave_multi_prefix_product(value, wide_mask);
  // expected-error@-1 {{passing 'uint64_t4'}}
  __builtin_hlsl_wave_multi_prefix_product(value, struct_mask);
  // expected-error@-1 {{passing 'S' to parameter of incompatible type}}
}
#endif
