// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -finclude-default-header -x hlsl -verify -o - %s
// RUN: %clang_cc1 -triple spirv-pc-vulkan1.3-library -finclude-default-header -x hlsl -verify -o - %s

float4 bad_uint(uint ff : SV_IsFrontFace) : SV_Target { return float4(ff); }
// expected-error@-1 {{attribute 'SV_IsFrontFace' only applies to a field or parameter of type 'bool'}}

float4 bad_float(float ff : SV_IsFrontFace) : SV_Target { return float4(ff); }
// expected-error@-1 {{attribute 'SV_IsFrontFace' only applies to a field or parameter of type 'bool'}}

float4 bad_int(int ff : SV_IsFrontFace) : SV_Target { return float4(ff); }
// expected-error@-1 {{attribute 'SV_IsFrontFace' only applies to a field or parameter of type 'bool'}}

float4 bad_vec(bool3 ff : SV_IsFrontFace) : SV_Target { return float4(ff.x); }
// expected-error@-1 {{attribute 'SV_IsFrontFace' only applies to a field or parameter of type 'bool'}}
//   `isBooleanType()` is scalar-only, so bool3 is rejected without an extra shape check.

// Indexing is not permitted on SV_IsFrontFace.
float4 bad_index(bool ff : SV_IsFrontFace1) : SV_Target { return float4(ff); }
// expected-error@-1 {{semantic 'SV_IsFrontFace' does not allow indexing}}

float4 ok(bool ff : SV_IsFrontFace) : SV_Target { return float4(ff); }
