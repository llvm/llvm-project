// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.6-library -finclude-default-header -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple spirv-pc-vulkan1.3-library -finclude-default-header -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefix=SPIRV

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

[shader("vertex")]
noperspective centroid Inner vertex() : RESULT {
  Inner result = {0, 0};
  return result;
}

// Inherit from entry parameters and return declarations, replacing the entire
// mask when an inner field supplies its own modifiers. Each use is independent.
// SPIRV-DAG: @A0 = {{.*}} !spirv.Decorations ![[A0:[0-9]+]]
// SPIRV-DAG: @A1 = {{.*}} !spirv.Decorations ![[A1:[0-9]+]]
// SPIRV-DAG: @A2 = {{.*}} !spirv.Decorations ![[A2:[0-9]+]]
// SPIRV-DAG: @A4 = {{.*}} !spirv.Decorations ![[A4:[0-9]+]]
// SPIRV-DAG: @B0 = {{.*}} !spirv.Decorations ![[B0:[0-9]+]]
// SPIRV-DAG: @C0 = {{.*}} !spirv.Decorations ![[C0:[0-9]+]]
// SPIRV-DAG: @SV_Position = {{.*}} !spirv.Decorations ![[POS:[0-9]+]]
// SPIRV-DAG: @RESULT0 = {{.*}} !spirv.Decorations ![[RESULT0:[0-9]+]]
// SPIRV-DAG: @RESULT1 = {{.*}} !spirv.Decorations ![[RESULT1:[0-9]+]]
// SPIRV-DAG: ![[A0]] = !{!{{[0-9]+}}, ![[FLAT:[0-9]+]]}
// SPIRV-DAG: ![[A1]] = !{!{{[0-9]+}}}
// SPIRV-DAG: ![[A2]] = !{!{{[0-9]+}}, ![[NP:[0-9]+]], ![[SAMPLE:[0-9]+]]}
// SPIRV-DAG: ![[A4]] = !{!{{[0-9]+}}, ![[FLAT]]}
// SPIRV-DAG: ![[B0]] = !{!{{[0-9]+}}, ![[CENTROID:[0-9]+]]}
// SPIRV-DAG: ![[C0]] = !{!{{[0-9]+}}}
// SPIRV-DAG: ![[POS]] = !{![[BUILTIN:[0-9]+]], ![[SAMPLE]]}
// SPIRV-DAG: ![[RESULT0]] = !{!{{[0-9]+}}, ![[NP]], ![[CENTROID]]}
// SPIRV-DAG: ![[RESULT1]] = !{!{{[0-9]+}}}
// SPIRV-DAG: ![[FLAT]] = !{i32 14}
// SPIRV-DAG: ![[NP]] = !{i32 13}
// SPIRV-DAG: ![[SAMPLE]] = !{i32 17}
// SPIRV-DAG: ![[CENTROID]] = !{i32 16}
// SPIRV-DAG: ![[BUILTIN]] = !{i32 11, i32 15}
