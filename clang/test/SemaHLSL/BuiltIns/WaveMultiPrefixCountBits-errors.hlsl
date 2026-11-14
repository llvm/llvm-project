// RUN: %clang_cc1 -finclude-default-header \
// RUN:   -triple dxil-pc-shadermodel6.5-library -verify %s
// RUN: %clang_cc1 -finclude-default-header -DTEST_AVAILABILITY \
// RUN:   -triple dxil-pc-shadermodel6.4-library -verify=availability %s
// RUN: %clang_cc1 -finclude-default-header -DTEST_UNSUPPORTED \
// RUN:   -triple spirv-pc-vulkan-library -verify=spirv %s

#ifdef TEST_AVAILABILITY
[shader("compute")]
[numthreads(1, 1, 1)]
void main() {
  (void)WaveMultiPrefixCountBits(true, uint4(1, 2, 4, 8));
  // availability-error@-1 {{only available on Shader Model 6.5 or newer}}
  // availability-note@*:* {{introduced in Shader Model 6.5}}
}
#elif defined(TEST_UNSUPPORTED)
uint test_public(bool value, uint4 mask) {
  return WaveMultiPrefixCountBits(value, mask);
  // spirv-error@-1 {{builtin is not supported on this target}}
}

uint test_builtin(bool value, uint4 mask) {
  return __builtin_hlsl_wave_multi_prefix_count_bits(value, mask);
  // spirv-error@-1 {{builtin is not supported on this target}}
}
#else
void test_arg_count(bool value, uint4 mask) {
  __builtin_hlsl_wave_multi_prefix_count_bits(value);
  // expected-error@-1 {{too few arguments to function call, expected 2}}
  __builtin_hlsl_wave_multi_prefix_count_bits(value, mask, mask);
  // expected-error@-1 {{too many arguments to function call, expected 2}}
}

struct S { uint value; };

void test_value_types(float f, int i, bool2 bv, S s, uint4 mask) {
  __builtin_hlsl_wave_multi_prefix_count_bits(f, mask);
  // expected-error@-1 {{invalid operand of type 'float'}}
  __builtin_hlsl_wave_multi_prefix_count_bits(i, mask);
  // expected-error@-1 {{invalid operand of type 'int'}}
  __builtin_hlsl_wave_multi_prefix_count_bits(bv, mask);
  // expected-error@-1 {{invalid operand of type 'bool2'}}
  __builtin_hlsl_wave_multi_prefix_count_bits(s, mask);
  // expected-error@-1 {{invalid operand of type 'S'}}
}

void test_mask_types(bool value, uint scalar, uint3 short_mask,
                     int4 signed_mask, float4 float_mask,
                     uint64_t4 wide_mask, S struct_mask) {
  __builtin_hlsl_wave_multi_prefix_count_bits(value, scalar);
  // expected-error@-1 {{passing 'uint' (aka 'unsigned int')}}
  __builtin_hlsl_wave_multi_prefix_count_bits(value, short_mask);
  // expected-error@-1 {{passing 'uint3' (aka 'vector<uint, 3>')}}
  __builtin_hlsl_wave_multi_prefix_count_bits(value, signed_mask);
  // expected-error@-1 {{passing 'int4' (aka 'vector<int, 4>')}}
  __builtin_hlsl_wave_multi_prefix_count_bits(value, float_mask);
  // expected-error@-1 {{passing 'float4' (aka 'vector<float, 4>')}}
  __builtin_hlsl_wave_multi_prefix_count_bits(value, wide_mask);
  // expected-error@-1 {{passing 'uint64_t4'}}
  __builtin_hlsl_wave_multi_prefix_count_bits(value, struct_mask);
  // expected-error@-1 {{passing 'S' to parameter of incompatible type}}
}
#endif
