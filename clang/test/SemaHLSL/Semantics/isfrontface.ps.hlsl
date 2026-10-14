// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -finclude-default-header -x hlsl -verify -o - %s
// RUN: %clang_cc1 -triple spirv-pc-vulkan1.3-library -finclude-default-header -x hlsl -verify -o - %s

[shader("pixel")]
float4 bad_uint(uint ff : SV_IsFrontFace) : SV_Target {
// expected-error@-1 {{semantic 'SV_IsFrontFace' must be a scalar of bool type (was 'uint' (aka 'unsigned int'))}}
  return ff ? float4(1, 0, 0, 1) : float4(0, 0, 1, 1);
}

[shader("pixel")]
float4 bad_float(float ff : SV_IsFrontFace) : SV_Target {
// expected-error@-1 {{semantic 'SV_IsFrontFace' must be a scalar of bool type (was 'float')}}
  return (ff.x > 0.f) ? float4(1, 0, 0, 1) : float4(0, 0, 1, 1);
}

[shader("pixel")]
float4 bad_vec(bool3 ff : SV_IsFrontFace) : SV_Target {
// expected-error@-1 {{semantic 'SV_IsFrontFace' must be a scalar of bool type (was 'bool3' (aka 'vector<bool, 3>'))}}
  return ff.x ? float4(1, 0, 0, 1) : float4(0, 0, 1, 1);
}

[shader("pixel")]
float4 bad_index(bool ff : SV_IsFrontFace1) : SV_Target {
// expected-error@-1 {{semantic 'SV_IsFrontFace' does not allow indexing}}
  return ff ? float4(1, 0, 0, 1) : float4(0, 0, 1, 1);
}

[shader("pixel")]
float4 ok(bool ff : SV_IsFrontFace) : SV_Target {
  return ff ? float4(1, 0, 0, 1) : float4(0, 0, 1, 1);
}
