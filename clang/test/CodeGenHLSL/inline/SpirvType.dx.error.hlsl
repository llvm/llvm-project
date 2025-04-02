// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.0-compute %s \
// RUN:   -fsyntax-only -verify

typedef vk::SpirvType<12, 2, 4, float> InvalidType1;  // expected-error {{use of undeclared identifier 'vk'}}
vk::Literal<nullptr> Unused;                          // expected-error {{use of undeclared identifier 'vk'}}
vk::integral_constant<uint, 456> Unused2;             // expected-error {{use of undeclared identifier 'vk'}}
typedef vk::SpirvOpaqueType<12, float> InvalidType2;  // expected-error {{use of undeclared identifier 'vk'}}

[numthreads(1, 1, 1)]
void main() {
}
