// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-vertex -x hlsl -emit-llvm -finclude-default-header -disable-llvm-passes -o - %s | FileCheck %s

// SV_Position is an arbitrary semantic on a vertex input, so its input
// signature element has the arbitrary semantic kind (0), while the output keeps
// the SV_Position kind (3).
float4 main(float4 P : SV_Position) : SV_Position {
  return P;
}

// CHECK-DAG: !{ptr @main, ![[#IN:]], ![[#OUT:]]}
// CHECK-DAG: ![[#IN]] = !{![[#IN_POS:]]}
// CHECK-DAG: ![[#IN_POS]] = !{i32 0, !"SV_Position", i32 9, i32 0, ![[#ZERO:]], i32 0, i32 1, i8 4, i32 0, i8 0, i8 0, i8 0, i32 0}
// CHECK-DAG: ![[#OUT]] = !{![[#OUT_POS:]]}
// CHECK-DAG: ![[#OUT_POS]] = !{i32 0, !"SV_Position", i32 9, i32 3, ![[#ZERO]], i32 4, i32 1, i8 4, i32 0, i8 0, i8 0, i8 0, i32 0}
