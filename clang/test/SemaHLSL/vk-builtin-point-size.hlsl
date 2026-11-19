// RUN: %clang_cc1 -finclude-default-header -triple spirv-pc-vulkan1.3-vertex -hlsl-entry legal -fsyntax-only -verify %s

// PointSize must be a float, and is only legal as a vertex shader output
// (plus other stages not exercised here).
struct Out {
  float4 pos : SV_Position;
  [[vk::builtin("PointSize")]] float size : PSIZE;
};

[shader("vertex")]
Out legal([[vk::location(0)]] float4 a : A) {
  Out o;
  o.pos = a;
  o.size = a.x;
  return o;
}

// expected-no-diagnostics
