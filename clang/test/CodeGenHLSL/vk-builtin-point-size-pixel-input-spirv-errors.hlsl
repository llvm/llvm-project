// REQUIRES: spirv-registered-target

// RUN: split-file %s %t
// RUN: %clang_cc1 -finclude-default-header -triple spirv-pc-vulkan1.3-pixel -hlsl-entry main -O3 -S -o /dev/null -verify %t/read.hlsl
// RUN: %clang_cc1 -finclude-default-header -triple spirv-pc-vulkan1.3-pixel -hlsl-entry main -O3 -S -o /dev/null -verify %t/helper.hlsl
// RUN: %clang_cc1 -finclude-default-header -triple spirv-pc-vulkan1.3-pixel -hlsl-entry main -O3 -S -o /dev/null -verify %t/scalar.hlsl
// RUN: %clang_cc1 -finclude-default-header -triple spirv-pc-vulkan1.3-pixel -hlsl-entry main -O0 -S -o /dev/null -verify %t/unread-O0.hlsl

// The SPIR-V backend reports a pixel shader PointSize input that is still used
// after optimization. The error does not carry a source location.

//--- read.hlsl
struct Shared {
  float a : A;
  [[vk::builtin("PointSize")]] float size : PSIZE;
};

// expected-error@* {{BuiltIn PointSize cannot be used in a Fragment entry point (VUID-PointSize-PointSize-04314)}}
[shader("pixel")]
float4 main(Shared input) : SV_Target {
  return input.size;
}

//--- helper.hlsl
struct Shared {
  float a : A;
  [[vk::builtin("PointSize")]] float size : PSIZE;
};

float readSize(Shared input) { return input.size; }

// expected-error@* {{BuiltIn PointSize cannot be used in a Fragment entry point (VUID-PointSize-PointSize-04314)}}
[shader("pixel")]
float4 main(Shared input) : SV_Target {
  return readSize(input);
}

//--- scalar.hlsl
// expected-error@* {{BuiltIn PointSize cannot be used in a Fragment entry point (VUID-PointSize-PointSize-04314)}}
[shader("pixel")]
float4 main([[vk::builtin("PointSize")]] float size : PSIZE) : SV_Target {
  return size;
}

//--- unread-O0.hlsl
// At -O0 the entry point copies every field of the input struct, so an unread
// field is still used.
struct Shared {
  float a : A;
  [[vk::builtin("PointSize")]] float size : PSIZE;
};

// expected-error@* {{BuiltIn PointSize cannot be used in a Fragment entry point (VUID-PointSize-PointSize-04314)}}
[shader("pixel")]
float4 main(Shared input) : SV_Target {
  return float4(input.a, 0, 0, 1);
}
