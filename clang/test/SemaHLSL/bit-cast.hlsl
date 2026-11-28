// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple dxil-pc-shadermodel6.6-library -fsyntax-only -verify %s
// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple spirv-unknown-vulkan-library -fsyntax-only -verify %s

// expected-no-diagnostics

using namespace hlsl;

float test_scalar(int V) {
  return bit_cast<float>(V);
}

float4 test_vector(int4 V) {
  return bit_cast<float>(V);
}

float2x3 test_matrix(int2x3 M) {
  return bit_cast<float>(M);
}

static_assert(bit_cast<float>(0x3f800000) == 1.0f, "");
