// RUN: %clang_cc1 -finclude-default-header -triple spirv-unknown-vulkan1.3-vertex -hlsl-entry main1 -fsyntax-only -verify %s

// Same as semantic.explicit-mix-vk-builtin-field.hlsl, but vk::builtin
// annotates the entry point's return value (the Function subject), mixed
// with an explicit output location on an `out` parameter.
[[vk::builtin("PointSize")]]
[shader("vertex")]
float main1(
    [[vk::location(0)]] float4 a : A, [[vk::location(1)]] out float4 b : B) : PSIZE {
  b = a;
  return a.x;
}

// expected-no-diagnostics
