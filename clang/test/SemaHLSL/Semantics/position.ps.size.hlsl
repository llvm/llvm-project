// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -finclude-default-header -o - %s -verify -verify-ignore-unexpected
// RUN: %clang_cc1 -triple spirv-unknown-vulkan1.3-library  -x hlsl -finclude-default-header -o - %s -verify -verify-ignore-unexpected

[shader("pixel")]
void too_many_components(vector<float, 5> a : SV_Position) {
// expected-error@-1 {{semantic 'SV_Position' must be a scalar or vector of up to 4 components of 16 or 32 bit floating-point type (was 'vector<float, 5>' (vector of 5 'float' values))}}
}

[shader("pixel")]
void not_a_float(int2 a : SV_Position) {
// expected-error@-1 {{semantic 'SV_Position' must be a scalar or vector of up to 4 components of 16 or 32 bit floating-point type (was 'int2' (aka 'vector<int, 2>'))}}
}

[shader("pixel")]
void too_wide(double4 a : SV_Position) {
// expected-error@-1 {{semantic 'SV_Position' must be a scalar or vector of up to 4 components of 16 or 32 bit floating-point type (was 'double4' (aka 'vector<double, 4>'))}}
}

[shader("pixel")]
void ok(float4 a : SV_Position) {
}
