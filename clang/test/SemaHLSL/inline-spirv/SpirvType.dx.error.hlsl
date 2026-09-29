// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.0-compute %s \
// RUN:   -fsyntax-only -verify

typedef vk::SpirvType<12, 2, 4, float> InvalidType; // expected-error {{use of undeclared identifier 'vk'}}

[numthreads(1, 1, 1)]
void main() {
    __hlsl_spirv_type<12, 2, 4, float> InvalidValue; // expected-error {{'__hlsl_spirv_type' is only available for the SPIR-V target}}
}
