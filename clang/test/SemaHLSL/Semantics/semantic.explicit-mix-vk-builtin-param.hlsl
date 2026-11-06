// RUN: %clang_cc1 -finclude-default-header -triple spirv-unknown-vulkan1.3-pixel -hlsl-entry main1 -fsyntax-only -verify %s

// Same as semantic.explicit-mix-vk-builtin-field.hlsl, but for a vk::builtin
// annotated entry-point parameter instead of a struct field.
// SPIRV-VAL-SKIP: PointSize as a pixel shader input is accepted by Sema, but
// Vulkan (VUID-PointSize-PointSize-04314) only allows BuiltIn PointSize in
// vertex, tessellation and geometry stages, so spirv-val rejects the module.
[shader("pixel")]
float4 main1([[vk::location(0)]] float4 a : A,
             [[vk::builtin("PointSize")]] float size : PSIZE) : SV_Target {
  return a * size;
}

// expected-no-diagnostics
