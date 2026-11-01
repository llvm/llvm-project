// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.6-library -finclude-default-header -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s

struct Inner {
  float inherited;
  linear float overridden;
};
struct Outer {
  Inner inherited;
  sample noperspective Inner overridden;
  float2 array[2][3];
};

[shader("pixel")]
float4 main(nointerpolation Outer a : A, centroid Outer b : B,
            Outer c : C) : SV_Target {
  return 0;
}

// Each use of Outer inherits independently. Inner's explicit linear replaces
// the entire inherited mask, and a field's modifier does not affect siblings.

// CHECK-DAG: !{i32 0,  !"A", i32 9, i32 0, !{{[0-9]+}}, i32 1, i32 1, i8 1,
// CHECK-DAG: !{i32 1,  !"A", i32 9, i32 0, !{{[0-9]+}}, i32 2, i32 1, i8 1,
// CHECK-DAG: !{i32 2,  !"A", i32 9, i32 0, !{{[0-9]+}}, i32 7, i32 1, i8 1,
// CHECK-DAG: !{i32 3,  !"A", i32 9, i32 0, !{{[0-9]+}}, i32 2, i32 1, i8 1,
// CHECK-DAG: !{i32 4,  !"A", i32 9, i32 0, !{{[0-9]+}}, i32 1, i32 6, i8 2,
// CHECK-DAG: !{i32 5,  !"B", i32 9, i32 0, !{{[0-9]+}}, i32 3, i32 1, i8 1,
// CHECK-DAG: !{i32 6,  !"B", i32 9, i32 0, !{{[0-9]+}}, i32 2, i32 1, i8 1,
// CHECK-DAG: !{i32 7,  !"B", i32 9, i32 0, !{{[0-9]+}}, i32 7, i32 1, i8 1,
// CHECK-DAG: !{i32 8,  !"B", i32 9, i32 0, !{{[0-9]+}}, i32 2, i32 1, i8 1,
// CHECK-DAG: !{i32 9,  !"B", i32 9, i32 0, !{{[0-9]+}}, i32 3, i32 6, i8 2,
// CHECK-DAG: !{i32 10, !"C", i32 9, i32 0, !{{[0-9]+}}, i32 2, i32 1, i8 1,
// CHECK-DAG: !{i32 11, !"C", i32 9, i32 0, !{{[0-9]+}}, i32 2, i32 1, i8 1,
// CHECK-DAG: !{i32 12, !"C", i32 9, i32 0, !{{[0-9]+}}, i32 7, i32 1, i8 1,
// CHECK-DAG: !{i32 13, !"C", i32 9, i32 0, !{{[0-9]+}}, i32 2, i32 1, i8 1,
// CHECK-DAG: !{i32 14, !"C", i32 9, i32 0, !{{[0-9]+}}, i32 2, i32 6, i8 2,
//                                                           ^ Interpolation mode.

struct Position {
  float4 p : SV_Position;
};
[shader("pixel")]
float4 position(sample Position p) : SV_Target { return p.p; }
// CHECK-DAG: !{i32 0, !"SV_Position", i32 9, i32 3, !{{[0-9]+}}, i32 7, i32 1, i8 4,
//                                                                    ^ Interpolation mode.
