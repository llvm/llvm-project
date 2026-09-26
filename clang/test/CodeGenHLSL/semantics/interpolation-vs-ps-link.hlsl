// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -emit-llvm -finclude-default-header -disable-llvm-passes -o - %s | FileCheck %s

struct Nested {
  float inherited : INHERIT;
  linear float overrideMode : OVERRIDE;
};

struct Varyings {
  float4 pos : SV_Position;
  float value : LIN;
  nointerpolation float flat : FLAT;
  noperspective float screen : NOPERS;
  centroid float cent : CENTRO;
  uint face : FACEID;
  nointerpolation Nested nested;
};

[shader("vertex")]
Varyings vs(float4 pos : POSITION, float value : VAL) {
  Varyings result;
  result.pos = pos;
  result.value = value;
  result.flat = value;
  result.screen = value;
  result.cent = value;
  result.face = 1;
  result.nested.inherited = value;
  result.nested.overrideMode = value;
  return result;
}

[shader("pixel")]
float4 ps(Varyings input) : SV_Target {
  return float4(input.value, input.flat, input.screen,
                input.nested.inherited + input.nested.overrideMode);
}

// Vertex outputs and pixel inputs must have identical packed signatures,
// including their interpolation modes; only then can the shaders link.
// CHECK: !dx.semantic.signatures = !{![[VS:[0-9]+]], ![[PS:[0-9]+]]}
// CHECK-DAG: ![[VS]] = !{ptr @vs, ![[ATTR:[0-9]+]], ![[VARY:[0-9]+]]}
// CHECK-DAG: ![[PS]] = !{ptr @ps, ![[VARY]], ![[TARGET:[0-9]+]]}
// CHECK-DAG: !{i32 0, !"SV_Position", i32 9, i32 3, !{{[0-9]+}}, i32 4, i32 1, i8 4,
// CHECK-DAG: !{i32 1, !"LIN", i32 9, i32 0, !{{[0-9]+}}, i32 2, i32 1, i8 1,
// CHECK-DAG: !{i32 2, !"FLAT", i32 9, i32 0, !{{[0-9]+}}, i32 1, i32 1, i8 1,
// CHECK-DAG: !{i32 3, !"NOPERS", i32 9, i32 0, !{{[0-9]+}}, i32 4, i32 1, i8 1,
// CHECK-DAG: !{i32 4, !"CENTRO", i32 9, i32 0, !{{[0-9]+}}, i32 3, i32 1, i8 1,
// CHECK-DAG: !{i32 5, !"FACEID", i32 5, i32 0, !{{[0-9]+}}, i32 1, i32 1, i8 1,
// CHECK-DAG: !{i32 6, !"INHERIT", i32 9, i32 0, !{{[0-9]+}}, i32 1, i32 1, i8 1,
// CHECK-DAG: !{i32 7, !"OVERRIDE", i32 9, i32 0, !{{[0-9]+}}, i32 2, i32 1, i8 1,
// Pixel outputs and vertex inputs do not use interpolation modes.
// CHECK-DAG: !{i32 0, !"POSITION", i32 9, i32 0, !{{[0-9]+}}, i32 0, i32 1, i8 4,
// CHECK-DAG: !{i32 0, !"SV_Target", i32 9, i32 16, !{{[0-9]+}}, i32 0, i32 1, i8 4,
