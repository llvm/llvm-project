// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.6-library -verify -fnative-half-type -fnative-int16-type %s
// RUN: %clang_cc1 -triple spirv-linux-vulkan-library -verify -fnative-half-type -fnative-int16-type %s

// expected-no-diagnostics
_Static_assert(sizeof(uint8_t4_packed) == 4, "uint8_t4_packed is 4 bytes");
