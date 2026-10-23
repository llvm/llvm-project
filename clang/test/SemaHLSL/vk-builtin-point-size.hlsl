// RUN: %clang_cc1 -finclude-default-header -triple spirv-pc-vulkan1.3-vertex -hlsl-entry legal -fsyntax-only -verify %s

// PointSize must be a float, and is only legal as a vertex shader output
// (plus other stages not exercised here).
[shader("vertex")]
float legal([[vk::location(0)]] float4 a : A,
           [[vk::builtin("PointSize")]] out float size : PSIZE) : B {
  size = a.x;
  return a.y;
}

// expected-no-diagnostics
