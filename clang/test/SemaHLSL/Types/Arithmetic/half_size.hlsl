// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.4-library -verify %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.4-library -verify -fnative-half-type %s
// RUN: %clang_cc1 -triple spirv-linux-vulkan-library -verify %s
// RUN: %clang_cc1 -triple spirv-linux-vulkan-library -verify -fnative-half-type %s

// expected-no-diagnostics
#ifdef __HLSL_ENABLE_16_BIT
_Static_assert(sizeof(half) == 2, "half is 2 bytes");
#else
_Static_assert(sizeof(half) == 4, "half is 4 bytes");
#endif
