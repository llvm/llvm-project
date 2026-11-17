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
  (void)WaveMultiPrefixBitAnd(1u, uint4(1, 2, 4, 8));
  // availability-error@-1 {{'WaveMultiPrefixBitAnd' is only available on Shader Model 6.5 or newer}}
  // availability-note@hlsl/hlsl_alias_intrinsics_gen.inc:* {{'WaveMultiPrefixBitAnd' has been marked as being introduced in Shader Model 6.5 here}}
}
#else
void test_arg_count(uint value, uint4 mask) {
  __builtin_hlsl_wave_multi_prefix_bit_and(value);
  // expected-error@-1 {{too few arguments to function call, expected 2, have 1}}
  __builtin_hlsl_wave_multi_prefix_bit_and(value, mask, mask);
  // expected-error@-1 {{too many arguments to function call, expected 2, have 3}}
}

struct S { uint value; };

void test_value_types(float f, float2 fv, bool b, bool2 bv, S s,
                      float2x2 fm, bool2x2 bm, uint4 mask) {
  __builtin_hlsl_wave_multi_prefix_bit_and(f, mask);
  // expected-error@-1 {{1st argument must be a scalar or vector of \
integer types (was 'double')}}
  __builtin_hlsl_wave_multi_prefix_bit_and(fv, mask);
  // expected-error@-1 {{1st argument must be a scalar or vector of \
integer types (was 'float2' (aka 'vector<float, 2>'))}}
  __builtin_hlsl_wave_multi_prefix_bit_and(b, mask);
  // expected-error@-1 {{invalid operand of type 'bool'}}
  __builtin_hlsl_wave_multi_prefix_bit_and(bv, mask);
  // expected-error@-1 {{invalid operand of type 'bool2'}}
  __builtin_hlsl_wave_multi_prefix_bit_and(s, mask);
  // expected-error@-1 {{invalid operand of type 'S'}}
  __builtin_hlsl_wave_multi_prefix_bit_and(fm, mask);
  // expected-error@-1 {{1st argument must be a matrix integer type \
(was 'float2x2' (aka 'matrix<float, 2, 2>'))}}
  // expected-error@-3 {{cannot pass object of non-trivial type 'float2x2'}}
  __builtin_hlsl_wave_multi_prefix_bit_and(bm, mask);
  // expected-error@-1 {{1st argument must be a matrix integer type \
(was 'bool2x2' (aka 'matrix<bool, 2, 2>'))}}
  // expected-error@-3 {{cannot pass object of non-trivial type 'bool2x2'}}
}

void test_mask_types(uint value, uint scalar, uint3 short_mask,
                     int4 signed_mask, float4 float_mask,
                     uint64_t4 wide_mask, S struct_mask) {
  __builtin_hlsl_wave_multi_prefix_bit_and(value, scalar);
  // expected-error@-1 {{passing 'uint' (aka 'unsigned int') to parameter of incompatible type 'vector<unsigned int, 4>'}}
  __builtin_hlsl_wave_multi_prefix_bit_and(value, short_mask);
  // expected-error@-1 {{passing 'uint3' (aka 'vector<uint, 3>') to parameter of incompatible type 'vector<unsigned int, 4>'}}
  __builtin_hlsl_wave_multi_prefix_bit_and(value, signed_mask);
  // expected-error@-1 {{passing 'int4' (aka 'vector<int, 4>') to parameter of incompatible type 'vector<unsigned int, 4>'}}
  __builtin_hlsl_wave_multi_prefix_bit_and(value, float_mask);
  // expected-error@-1 {{passing 'float4' (aka 'vector<float, 4>') to parameter of incompatible type 'vector<unsigned int, 4>'}}
  __builtin_hlsl_wave_multi_prefix_bit_and(value, wide_mask);
  // expected-error@-1 {{passing 'uint64_t4'}}
  __builtin_hlsl_wave_multi_prefix_bit_and(value, struct_mask);
  // expected-error@-1 {{passing 'S' to parameter of incompatible type 'vector<unsigned int, 4>'}}
}
#endif
