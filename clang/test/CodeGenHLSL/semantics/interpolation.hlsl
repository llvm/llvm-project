// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.6-library -finclude-default-header -Wno-ignored-attributes -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s

[shader("pixel")]
float4 modes(float a : A,
             nointerpolation float b : B,
             linear float c : C,
             centroid float d : D,
             noperspective float e : E,
             noperspective centroid float f : F,
             sample float g : G,
             noperspective sample float h : H,
             center float i : I,
             linear center float j : J,
             linear centroid float k : K,
             linear noperspective center float l : L,
             linear noperspective centroid float m : M,
             linear sample float n : N,
             sample linear noperspective float o : O) : SV_Target {
  return 0;
}

// CHECK-DAG: !{i32 0,  !"A", i32 9, i32 0, !{{[0-9]+}}, i32 2, i32 1, i8 1,
// CHECK-DAG: !{i32 1,  !"B", i32 9, i32 0, !{{[0-9]+}}, i32 1, i32 1, i8 1,
// CHECK-DAG: !{i32 2,  !"C", i32 9, i32 0, !{{[0-9]+}}, i32 2, i32 1, i8 1,
// CHECK-DAG: !{i32 3,  !"D", i32 9, i32 0, !{{[0-9]+}}, i32 3, i32 1, i8 1,
// CHECK-DAG: !{i32 4,  !"E", i32 9, i32 0, !{{[0-9]+}}, i32 4, i32 1, i8 1,
// CHECK-DAG: !{i32 5,  !"F", i32 9, i32 0, !{{[0-9]+}}, i32 5, i32 1, i8 1,
// CHECK-DAG: !{i32 6,  !"G", i32 9, i32 0, !{{[0-9]+}}, i32 6, i32 1, i8 1,
// CHECK-DAG: !{i32 7,  !"H", i32 9, i32 0, !{{[0-9]+}}, i32 7, i32 1, i8 1,
// CHECK-DAG: !{i32 8,  !"I", i32 9, i32 0, !{{[0-9]+}}, i32 2, i32 1, i8 1,
// CHECK-DAG: !{i32 9,  !"J", i32 9, i32 0, !{{[0-9]+}}, i32 2, i32 1, i8 1,
// CHECK-DAG: !{i32 10, !"K", i32 9, i32 0, !{{[0-9]+}}, i32 3, i32 1, i8 1,
// CHECK-DAG: !{i32 11, !"L", i32 9, i32 0, !{{[0-9]+}}, i32 4, i32 1, i8 1,
// CHECK-DAG: !{i32 12, !"M", i32 9, i32 0, !{{[0-9]+}}, i32 5, i32 1, i8 1,
// CHECK-DAG: !{i32 13, !"N", i32 9, i32 0, !{{[0-9]+}}, i32 6, i32 1, i8 1,
// CHECK-DAG: !{i32 14, !"O", i32 9, i32 0, !{{[0-9]+}}, i32 7, i32 1, i8 1,
//                                                           ^ Interpolation mode.

[shader("pixel")]
sample float4 defaults(int a : INT, uint2 b : UINT, bool c : BOOL,
                       nointerpolation float2 e : FLAT, half f : HALF)
    : SV_Target {
  return 0;
}
// CHECK-DAG: !{i32 0, !"INT",    i32 4,          i32 0, !{{[0-9]+}}, i32 1, i32 1, i8 1,
// CHECK-DAG: !{i32 1, !"UINT",   i32 5,          i32 0, !{{[0-9]+}}, i32 1, i32 1, i8 2,
// CHECK-DAG: !{i32 2, !"BOOL",   i32 {{[0-9]+}}, i32 0, !{{[0-9]+}}, i32 1, i32 1, i8 1,
// CHECK-DAG: !{i32 3, !"FLAT",   i32 9,          i32 0, !{{[0-9]+}}, i32 1, i32 1, i8 2,
// CHECK-DAG: !{i32 4, !"HALF",   i32 {{[89]}},   i32 0, !{{[0-9]+}}, i32 2, i32 1, i8 1,
//                                                                        ^ Interpolation mode.
// Pixel outputs remain Undefined, including explicitly qualified returns.
// CHECK-DAG: !{i32 0, !"SV_Target", i32 9, i32 16, !{{[0-9]+}}, i32 0, i32 1, i8 4,
//                                                                   ^ Interpolation mode.

[shader("pixel")]
float4 position_default(float4 a : SV_Position) : SV_Target {
  return a;
}
// CHECK-DAG: !{ptr @position_default, ![[DEFSIG:[0-9]+]], !{{[0-9]+}}}
// CHECK-DAG: ![[DEFSIG]] = !{![[DEFPOS:[0-9]+]]}
// CHECK-DAG: ![[DEFPOS]] = !{i32 0, !"SV_Position", i32 9, i32 3, !{{[0-9]+}}, i32 4, i32 1, i8 4,

[shader("pixel")]
float4 position_centroid(centroid float4 a : SV_Position) : SV_Target {
  return a;
}
// CHECK-DAG: !{ptr @position_centroid, ![[CENTSIG:[0-9]+]], !{{[0-9]+}}}
// CHECK-DAG: ![[CENTSIG]] = !{![[CENTPOS:[0-9]+]]}
// CHECK-DAG: ![[CENTPOS]] = !{i32 0, !"SV_Position", i32 9, i32 3, !{{[0-9]+}}, i32 5, i32 1, i8 4,

// 'sample' and 'noperspective sample' produce the same signature, since
// SV_Position is always noperspective.

[shader("pixel")]
float4 position_sample(sample float4 a : SV_Position) : SV_Target {
  return a;
}

[shader("pixel")]
float4 position_noperspective_sample(noperspective sample float4 a
                                     : SV_Position) : SV_Target {
  return a;
}
// CHECK-DAG: !{ptr @position_sample, ![[SMPSIG:[0-9]+]], !{{[0-9]+}}}
// CHECK-DAG: !{ptr @position_noperspective_sample, ![[SMPSIG]], !{{[0-9]+}}}
// CHECK-DAG: ![[SMPSIG]] = !{![[SMPPOS:[0-9]+]]}
// CHECK-DAG: ![[SMPPOS]] = !{i32 0, !"SV_Position", i32 9, i32 3, !{{[0-9]+}}, i32 7, i32 1, i8 4,
//                                                                      ^ Interpolation mode.

[shader("pixel")]
float4 precedence(center centroid float a : CENTER_CENTROID,
                  centroid center float b : CENTROID_CENTER,
                  center sample float c : CENTER_SAMPLE,
                  sample center float d : SAMPLE_CENTER,
                  centroid sample float e : CENTROID_SAMPLE,
                  sample centroid float f : SAMPLE_CENTROID) : SV_Target {
  return 0;
}

// Sampling-location precedence is independent of keyword order.

// CHECK-DAG: !{i32 0, !"CENTER_CENTROID", i32 9, i32 0, !{{[0-9]+}}, i32 3, i32 1, i8 1,
// CHECK-DAG: !{i32 1, !"CENTROID_CENTER", i32 9, i32 0, !{{[0-9]+}}, i32 3, i32 1, i8 1,
// CHECK-DAG: !{i32 2, !"CENTER_SAMPLE",   i32 9, i32 0, !{{[0-9]+}}, i32 6, i32 1, i8 1,
// CHECK-DAG: !{i32 3, !"SAMPLE_CENTER",   i32 9, i32 0, !{{[0-9]+}}, i32 6, i32 1, i8 1,
// CHECK-DAG: !{i32 4, !"CENTROID_SAMPLE", i32 9, i32 0, !{{[0-9]+}}, i32 6, i32 1, i8 1,
// CHECK-DAG: !{i32 5, !"SAMPLE_CENTROID", i32 9, i32 0, !{{[0-9]+}}, i32 6, i32 1, i8 1,
//                                                                        ^ Interpolation mode.

[shader("vertex")]
centroid float4 vertex(sample float4 a : VERTEX,
                       nointerpolation uint b : SV_VertexID) : SV_Position {
  return a;
}

// Vertex inputs remain Undefined; vertex outputs carry the interpolation mode
// used to pack against pixel inputs. SV_Position is noperspective.
// CHECK-DAG: !{i32 0, !"VERTEX",      i32 9, i32 0, !{{[0-9]+}}, i32 0, i32 1, i8 4,
// CHECK-DAG: !{i32 1, !"SV_VertexID", i32 5, i32 1, !{{[0-9]+}}, i32 0, i32 1, i8 1,
// The vertex output signature is the centroid SV_Position signature above.
// CHECK-DAG: !{ptr @vertex, !{{[0-9]+}}, ![[CENTSIG]]}
