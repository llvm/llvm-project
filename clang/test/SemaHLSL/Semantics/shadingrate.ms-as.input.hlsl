// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.6-library -finclude-default-header -x hlsl -verify -o - %s
// RUN: %clang_cc1 -triple spirv-pc-vulkan1.3-library -finclude-default-header -x hlsl -verify -o - %s

// SV_ShadingRate is not an input in mesh or amplification shaders

[shader("mesh")]
[numthreads(1, 1, 1)]
void ms_in(uint r : SV_ShadingRate) {
// expected-error@-1 {{semantic 'SV_ShadingRate' is not supported in mesh shader inputs}}
}

[shader("amplification")]
[numthreads(1, 1, 1)]
void as_in(uint r : SV_ShadingRate) {
// expected-error@-1 {{semantic 'SV_ShadingRate' is not supported in amplification shader inputs}}
}
