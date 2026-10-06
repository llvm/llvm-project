// Per-row store.output emission for the range of semantic leaf types:
//   float          - scalar        -> 1 row
//   float4         - vector        -> 1 row (4 columns)
//   float[5]       - scalar array  -> 5 rows
//   float4[2][3]   - vector array  -> 6 rows (multidimensional, 4 columns each)
//
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -emit-llvm -finclude-default-header -disable-llvm-passes -o - %s | FileCheck %s

struct S {
  float a       : A;
  float4 b      : B;
  float d[5]    : D;
  float4 e[2][3] : E;
};

[shader("vertex")]
S main() {
  S s;
  return s;
}

// float a : A -> 1 row, 1 column.
// CHECK: %[[A:.*]] = extractvalue %struct.S %[[S:.*]], 0
// CHECK: call void @llvm.dx.store.output.f32(i32 0, i32 0, i8 0, float %[[A]])

// float4 b : B -> 1 row, 4 columns; the whole vector is stored.
// CHECK: %[[B:.*]] = extractvalue %struct.S %[[S]], 1
// CHECK: call void @llvm.dx.store.output.v4f32(i32 1, i32 0, i8 0, <4 x float> %[[B]])

// float d[5] : D -> 5 rows, 1 column each.
// CHECK: %[[D:.*]] = extractvalue %struct.S %[[S]], 2
// CHECK: %[[D0:.*]] = extractvalue [5 x float] %[[D]], 0
// CHECK: call void @llvm.dx.store.output.f32(i32 2, i32 0, i8 0, float %[[D0]])
// CHECK: %[[D1:.*]] = extractvalue [5 x float] %[[D]], 1
// CHECK: call void @llvm.dx.store.output.f32(i32 2, i32 1, i8 0, float %[[D1]])
// CHECK: %[[D2:.*]] = extractvalue [5 x float] %[[D]], 2
// CHECK: call void @llvm.dx.store.output.f32(i32 2, i32 2, i8 0, float %[[D2]])
// CHECK: %[[D3:.*]] = extractvalue [5 x float] %[[D]], 3
// CHECK: call void @llvm.dx.store.output.f32(i32 2, i32 3, i8 0, float %[[D3]])
// CHECK: %[[D4:.*]] = extractvalue [5 x float] %[[D]], 4
// CHECK: call void @llvm.dx.store.output.f32(i32 2, i32 4, i8 0, float %[[D4]])

// float4 e[2][3] : E -> 6 rows (2 x 3), 4 columns each; row-major flattening.
// CHECK: %[[E:.*]] = extractvalue %struct.S %[[S]], 3
// CHECK: %[[E00:.*]] = extractvalue [2 x [3 x <4 x float>]] %[[E]], 0, 0
// CHECK: call void @llvm.dx.store.output.v4f32(i32 3, i32 0, i8 0, <4 x float> %[[E00]])
// CHECK: %[[E01:.*]] = extractvalue [2 x [3 x <4 x float>]] %[[E]], 0, 1
// CHECK: call void @llvm.dx.store.output.v4f32(i32 3, i32 1, i8 0, <4 x float> %[[E01]])
// CHECK: %[[E02:.*]] = extractvalue [2 x [3 x <4 x float>]] %[[E]], 0, 2
// CHECK: call void @llvm.dx.store.output.v4f32(i32 3, i32 2, i8 0, <4 x float> %[[E02]])
// CHECK: %[[E10:.*]] = extractvalue [2 x [3 x <4 x float>]] %[[E]], 1, 0
// CHECK: call void @llvm.dx.store.output.v4f32(i32 3, i32 3, i8 0, <4 x float> %[[E10]])
// CHECK: %[[E11:.*]] = extractvalue [2 x [3 x <4 x float>]] %[[E]], 1, 1
// CHECK: call void @llvm.dx.store.output.v4f32(i32 3, i32 4, i8 0, <4 x float> %[[E11]])
// CHECK: %[[E12:.*]] = extractvalue [2 x [3 x <4 x float>]] %[[E]], 1, 2
// CHECK: call void @llvm.dx.store.output.v4f32(i32 3, i32 5, i8 0, <4 x float> %[[E12]])

// CHECK: !dx.semantic.signatures = !{![[#ENTRY_SIG:]]}
// CHECK: ![[#ENTRY_SIG]] = !{ptr @main, null, ![[#OUTPUT_SIG:]]}
// CHECK: ![[#OUTPUT_SIG]] = !{![[#A_SIG:]], ![[#B_SIG:]], ![[#D_SIG:]], ![[#E_SIG:]]}
// CHECK: ![[#A_SIG]] = !{i32 0, !"A", i32 9, i32 0, ![[#ZERO_INDEX:]], i32 0, i32 1, i8 1, i32 -1, i8 -1, i8 0, i8 0, i32 0}
// CHECK: ![[#ZERO_INDEX]] = !{i32 0}
// CHECK: ![[#B_SIG]] = !{i32 1, !"B", i32 9, i32 0, ![[#ZERO_INDEX]], i32 0, i32 1, i8 4, i32 -1, i8 -1, i8 0, i8 0, i32 0}
// CHECK: ![[#D_SIG]] = !{i32 2, !"D", i32 9, i32 0, ![[#D_INDICES:]], i32 0, i32 5, i8 1, i32 -1, i8 -1, i8 0, i8 0, i32 0}
// CHECK: ![[#D_INDICES]] = !{i32 0, i32 1, i32 2, i32 3, i32 4}
// CHECK: ![[#E_SIG]] = !{i32 3, !"E", i32 9, i32 0, ![[#E_INDICES:]], i32 0, i32 6, i8 4, i32 -1, i8 -1, i8 0, i8 0, i32 0}
// CHECK: ![[#E_INDICES]] = !{i32 0, i32 1, i32 2, i32 3, i32 4, i32 5}
