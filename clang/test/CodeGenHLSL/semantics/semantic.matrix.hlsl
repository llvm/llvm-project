// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -emit-llvm -finclude-default-header -disable-llvm-passes -fmatrix-memory-layout=row-major -o - %s | FileCheck %s --check-prefixes=CHECK,ROW
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -emit-llvm -finclude-default-header -disable-llvm-passes -fmatrix-memory-layout=column-major -o - %s | FileCheck %s --check-prefixes=CHECK,COLUMN

// Explicit layout overrides the default. A row-major 2x3 matrix occupies two
// signature rows, but direct arguments and returns are column-major vectors.
[shader("vertex")]
row_major float2x3 rowMatrix(row_major float2x3 m : RIN2) : ROUT4 {
  return m;
}

// CHECK-LABEL: define void @rowMatrix()
// CHECK: %[[R0:.*]] = call <3 x float> @llvm.dx.load.input.v3f32(i32 0, i32 0, i8 0, i32 poison)
// CHECK: %[[R00:.*]] = extractelement <3 x float> %[[R0]], i64 0
// CHECK: %[[R1:.*]] = insertelement <6 x float> poison, float %[[R00]], i64 0
// CHECK: %[[R01:.*]] = extractelement <3 x float> %[[R0]], i64 1
// CHECK: %[[R2:.*]] = insertelement <6 x float> %[[R1]], float %[[R01]], i64 2
// CHECK: %[[R02:.*]] = extractelement <3 x float> %[[R0]], i64 2
// CHECK: %[[R3:.*]] = insertelement <6 x float> %[[R2]], float %[[R02]], i64 4
// CHECK: %[[R4:.*]] = call <3 x float> @llvm.dx.load.input.v3f32(i32 0, i32 1, i8 0, i32 poison)
// CHECK: %[[R10:.*]] = extractelement <3 x float> %[[R4]], i64 0
// CHECK: %[[R5:.*]] = insertelement <6 x float> %[[R3]], float %[[R10]], i64 1
// CHECK: %[[R11:.*]] = extractelement <3 x float> %[[R4]], i64 1
// CHECK: %[[R6:.*]] = insertelement <6 x float> %[[R5]], float %[[R11]], i64 3
// CHECK: %[[R12:.*]] = extractelement <3 x float> %[[R4]], i64 2
// CHECK: %[[R7:.*]] = insertelement <6 x float> %[[R6]], float %[[R12]], i64 5
// CHECK: %[[RR:.*]] = call <6 x float> @{{.*}}rowMatrix{{.*}}(<6 x float> %[[R7]])
// CHECK: %[[RO0:.*]] = shufflevector <6 x float> %[[RR]], <6 x float> poison, <3 x i32> <i32 0, i32 2, i32 4>
// CHECK: call void @llvm.dx.store.output.v3f32(i32 0, i32 0, i8 0, <3 x float> %[[RO0]])
// CHECK: %[[RO1:.*]] = shufflevector <6 x float> %[[RR]], <6 x float> poison, <3 x i32> <i32 1, i32 3, i32 5>
// CHECK: call void @llvm.dx.store.output.v3f32(i32 0, i32 1, i8 0, <3 x float> %[[RO1]])

[shader("vertex")]
column_major float2x3 columnMatrix(column_major float2x3 m : CIN3) : COUT6 {
  return m;
}

// CHECK-LABEL: define void @columnMatrix()
// CHECK: call <2 x float> @llvm.dx.load.input.v2f32(i32 0, i32 0, i8 0, i32 poison)
// CHECK: insertelement <6 x float> poison, float %{{.*}}, i64 0
// CHECK: insertelement <6 x float> %{{.*}}, float %{{.*}}, i64 1
// CHECK: call <2 x float> @llvm.dx.load.input.v2f32(i32 0, i32 1, i8 0, i32 poison)
// CHECK: insertelement <6 x float> %{{.*}}, float %{{.*}}, i64 2
// CHECK: insertelement <6 x float> %{{.*}}, float %{{.*}}, i64 3
// CHECK: call <2 x float> @llvm.dx.load.input.v2f32(i32 0, i32 2, i8 0, i32 poison)
// CHECK: insertelement <6 x float> %{{.*}}, float %{{.*}}, i64 4
// CHECK: insertelement <6 x float> %{{.*}}, float %{{.*}}, i64 5
// CHECK: %[[CR:.*]] = call <6 x float> @{{.*}}columnMatrix
// CHECK: %[[CO0:.*]] = shufflevector <6 x float> %[[CR]], <6 x float> poison, <2 x i32> <i32 0, i32 1>
// CHECK: call void @llvm.dx.store.output.v2f32(i32 0, i32 0, i8 0, <2 x float> %[[CO0]])
// CHECK: %[[CO1:.*]] = shufflevector <6 x float> %[[CR]], <6 x float> poison, <2 x i32> <i32 2, i32 3>
// CHECK: call void @llvm.dx.store.output.v2f32(i32 0, i32 1, i8 0, <2 x float> %[[CO1]])
// CHECK: %[[CO2:.*]] = shufflevector <6 x float> %[[CR]], <6 x float> poison, <2 x i32> <i32 4, i32 5>
// CHECK: call void @llvm.dx.store.output.v2f32(i32 0, i32 2, i8 0, <2 x float> %[[CO2]])

[shader("vertex")]
float2x3 defaultMatrix(float2x3 m : DIN) : DOUT {
  return m;
}

// CHECK-LABEL: define void @defaultMatrix()
// ROW: call <3 x float> @llvm.dx.load.input.v3f32(i32 0, i32 1, i8 0, i32 poison)
// ROW-NOT: @llvm.dx.load.input
// ROW: call void @llvm.dx.store.output.v3f32(i32 0, i32 1, i8 0,
// COLUMN: call <2 x float> @llvm.dx.load.input.v2f32(i32 0, i32 2, i8 0, i32 poison)
// COLUMN-NOT: @llvm.dx.load.input
// COLUMN: call void @llvm.dx.store.output.v2f32(i32 0, i32 2, i8 0,

// Matrix rows extend the surrounding array dimensions. Aggregate matrices use
// their memory representation; bool components remain i32 in that representation.
struct Matrices {
  row_major float2x3 r[2][2] : ARRAY3;
  column_major bool2x3 b : BOOL9;
  row_major int1x3 singleRow : SINGLE7;
  column_major float3x1 singleColumn : COL6;
};

[shader("vertex")]
Matrices aggregateMatrix(Matrices m) {
  return m;
}

// CHECK-LABEL: define void @aggregateMatrix()
// CHECK: %[[AR0:.*]] = call <3 x float> @llvm.dx.load.input.v3f32(i32 0, i32 0, i8 0, i32 poison)
// CHECK: insertvalue [2 x [2 x [2 x <3 x float>]]] poison, <3 x float> %[[AR0]], 0, 0, 0
// CHECK: %[[AR7:.*]] = call <3 x float> @llvm.dx.load.input.v3f32(i32 0, i32 7, i8 0, i32 poison)
// CHECK: insertvalue [2 x [2 x [2 x <3 x float>]]] %{{.*}}, <3 x float> %[[AR7]], 1, 1, 1
// CHECK: %[[AB0:.*]] = call <2 x i32> @llvm.dx.load.input.v2i32(i32 1, i32 0, i8 0, i32 poison)
// CHECK: insertvalue [3 x <2 x i32>] poison, <2 x i32> %[[AB0]], 0
// CHECK: %[[AB2:.*]] = call <2 x i32> @llvm.dx.load.input.v2i32(i32 1, i32 2, i8 0, i32 poison)
// CHECK: insertvalue [3 x <2 x i32>] %{{.*}}, <2 x i32> %[[AB2]], 2
// CHECK: call <3 x i32> @llvm.dx.load.input.v3i32(i32 2, i32 0, i8 0, i32 poison)
// CHECK: call <3 x float> @llvm.dx.load.input.v3f32(i32 3, i32 0, i8 0, i32 poison)
// CHECK: %[[AO0:.*]] = extractvalue [2 x [2 x [2 x <3 x float>]]] %{{.*}}, 0, 0, 0
// CHECK: call void @llvm.dx.store.output.v3f32(i32 0, i32 0, i8 0, <3 x float> %[[AO0]])
// CHECK: %[[AO7:.*]] = extractvalue [2 x [2 x [2 x <3 x float>]]] %{{.*}}, 1, 1, 1
// CHECK: call void @llvm.dx.store.output.v3f32(i32 0, i32 7, i8 0, <3 x float> %[[AO7]])
// CHECK: %[[BO2:.*]] = extractvalue [3 x <2 x i32>] %{{.*}}, 2
// CHECK: call void @llvm.dx.store.output.v2i32(i32 1, i32 2, i8 0, <2 x i32> %[[BO2]])

[shader("vertex")]
row_major bool2x3 boolMatrix(row_major bool2x3 m : BIN) : BOUT {
  return m;
}

// CHECK-LABEL: define void @boolMatrix()
// CHECK: %[[BI0:.*]] = call <3 x i32> @llvm.dx.load.input.v3i32(i32 0, i32 0, i8 0, i32 poison)
// CHECK: icmp ne <3 x i32> %[[BI0]], zeroinitializer
// CHECK: %[[BI1:.*]] = call <3 x i32> @llvm.dx.load.input.v3i32(i32 0, i32 1, i8 0, i32 poison)
// CHECK: icmp ne <3 x i32> %[[BI1]], zeroinitializer
// CHECK: %[[BR:.*]] = call <6 x i1> @{{.*}}boolMatrix
// CHECK: %[[BS0:.*]] = shufflevector <6 x i1> %[[BR]], <6 x i1> poison, <3 x i32> <i32 0, i32 2, i32 4>
// CHECK: %[[BZ0:.*]] = zext <3 x i1> %[[BS0]] to <3 x i32>
// CHECK: call void @llvm.dx.store.output.v3i32(i32 0, i32 0, i8 0, <3 x i32> %[[BZ0]])
// CHECK: %[[BS1:.*]] = shufflevector <6 x i1> %[[BR]], <6 x i1> poison, <3 x i32> <i32 1, i32 3, i32 5>
// CHECK: %[[BZ1:.*]] = zext <3 x i1> %[[BS1]] to <3 x i32>
// CHECK: call void @llvm.dx.store.output.v3i32(i32 0, i32 1, i8 0, <3 x i32> %[[BZ1]])

// Check component types, row/column counts, and consecutive semantic indices.
// CHECK-DAG: !{i32 0, !"RIN", i32 9, i32 0, ![[RI:[0-9]+]], i32 0, i32 2, i8 3,
// CHECK-DAG: ![[RI]] = !{i32 2, i32 3}
// CHECK-DAG: !{i32 0, !"ROUT", i32 9, i32 0, ![[RO:[0-9]+]], i32 2, i32 2, i8 3,
// CHECK-DAG: ![[RO]] = !{i32 4, i32 5}
// CHECK-DAG: !{i32 0, !"CIN", i32 9, i32 0, ![[CI:[0-9]+]], i32 0, i32 3, i8 2,
// CHECK-DAG: ![[CI]] = !{i32 3, i32 4, i32 5}
// CHECK-DAG: !{i32 0, !"COUT", i32 9, i32 0, ![[CO:[0-9]+]], i32 2, i32 3, i8 2,
// CHECK-DAG: ![[CO]] = !{i32 6, i32 7, i32 8}
// ROW-DAG: !{i32 0, !"DIN", i32 9, i32 0, !{{[0-9]+}}, i32 0, i32 2, i8 3,
// ROW-DAG: !{i32 0, !"DOUT", i32 9, i32 0, !{{[0-9]+}}, i32 2, i32 2, i8 3,
// COLUMN-DAG: !{i32 0, !"DIN", i32 9, i32 0, !{{[0-9]+}}, i32 0, i32 3, i8 2,
// COLUMN-DAG: !{i32 0, !"DOUT", i32 9, i32 0, !{{[0-9]+}}, i32 2, i32 3, i8 2,
// CHECK-DAG: !{i32 0, !"ARRAY", i32 9, i32 0, ![[AI:[0-9]+]], i32 0, i32 8, i8 3,
// CHECK-DAG: !{i32 0, !"ARRAY", i32 9, i32 0, ![[AI]], i32 2, i32 8, i8 3,
// CHECK-DAG: ![[AI]] = !{i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10}
// CHECK-DAG: !{i32 1, !"BOOL", i32 5, i32 0, ![[BI:[0-9]+]], i32 0, i32 3, i8 2,
// CHECK-DAG: !{i32 1, !"BOOL", i32 5, i32 0, ![[BI]], i32 1, i32 3, i8 2,
// CHECK-DAG: ![[BI]] = !{i32 9, i32 10, i32 11}
// CHECK-DAG: !{i32 2, !"SINGLE", i32 4, i32 0, !{{[0-9]+}}, i32 0, i32 1, i8 3,
// CHECK-DAG: !{i32 2, !"SINGLE", i32 4, i32 0, !{{[0-9]+}}, i32 1, i32 1, i8 3,
// CHECK-DAG: !{i32 3, !"COL", i32 9, i32 0, !{{[0-9]+}}, i32 0, i32 1, i8 3,
// CHECK-DAG: !{i32 3, !"COL", i32 9, i32 0, !{{[0-9]+}}, i32 2, i32 1, i8 3,

// Implicit semantic indices continue after every matrix row.
struct ImplicitIndices {
  row_major float2x3 r;
  column_major float2x3 c;
  float tail;
};

[shader("vertex")]
float implicitMatrixIndex(ImplicitIndices input : TEXCOORD0) : OUT {
  return input.tail;
}

// CHECK-DAG: !{i32 0, !"TEXCOORD", i32 9, i32 0, ![[TR:[0-9]+]], i32 0, i32 2, i8 3,
// CHECK-DAG: ![[TR]] = !{i32 0, i32 1}
// CHECK-DAG: !{i32 1, !"TEXCOORD", i32 9, i32 0, ![[TC:[0-9]+]], i32 0, i32 3, i8 2,
// CHECK-DAG: ![[TC]] = !{i32 2, i32 3, i32 4}
// CHECK-DAG: !{i32 2, !"TEXCOORD", i32 9, i32 0, ![[TT:[0-9]+]], i32 0, i32 1, i8 1,
// CHECK-DAG: ![[TT]] = !{i32 5}
