// RUN: %clang_cc1 -finclude-default-header -triple spirv-unknown-vulkan1.3-vertex -hlsl-entry main1 -fsyntax-only -verify %s

// Same as semantic.explicit-mix-vk-builtin-field.hlsl, but for a vk::builtin
// annotated entry-point parameter instead of a struct field.
// SPIRV-VAL-SKIP: the `out` entry parameters are not supported by clang's
// SPIR-V CodeGen yet (llvm_unreachable "Not handled yet" in CGHLSLRuntime.cpp
// for parameters with HLSLParamModifierAttr). This test only covers Sema.
[shader("vertex")]
float4 main1([[vk::location(0)]] float4 a : A,
             [[vk::location(0)]] out float4 b : B,
             [[vk::builtin("PointSize")]] out float size : PSIZE) : SV_Position {
  b = a;
  size = a.x;
  return a;
}

// expected-no-diagnostics
