// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -disable-llvm-passes -emit-llvm -finclude-default-header -o - -fmatrix-memory-layout=column-major %s | FileCheck %s --check-prefixes=CHECK,COL-CHECK
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -disable-llvm-passes -emit-llvm -finclude-default-header -o - -fmatrix-memory-layout=row-major %s | FileCheck %s --check-prefixes=CHECK,ROW-CHECK

// CHECK-LABEL: define hidden noundef nofpclass(nan inf) <4 x float> @_Z2fnu11matrix_typeILm2ELm2EfE(
// CHECK-SAME: <4 x float> noundef nofpclass(nan inf) [[M:%.*]]) #[[ATTR0:[0-9]+]] {
// CHECK-NEXT:  [[ENTRY:.*:]]
// CHECK-NEXT:    [[M_ADDR:%.*]] = alloca [2 x <2 x float>], align 4
// CHECK-NEXT:    [[V:%.*]] = alloca <4 x float>, align 16
// CHECK-NEXT:    store <4 x float> [[M]], ptr [[M_ADDR]], align 4
// CHECK-NEXT:    [[TMP0:%.*]] = load <4 x float>, ptr [[M_ADDR]], align 4
// CHECK-NEXT:    [[MATRIXEXT:%.*]] = extractelement <4 x float> [[TMP0]], i32 0
// CHECK-NEXT:    [[VECINIT:%.*]] = insertelement <4 x float> poison, float [[MATRIXEXT]], i32 0
// CHECK-NEXT:    [[TMP1:%.*]] = load <4 x float>, ptr [[M_ADDR]], align 4
// COL-CHECK-NEXT:    [[MATRIXEXT1:%.*]] = extractelement <4 x float> [[TMP1]], i32 2
// ROW-CHECK-NEXT:    [[MATRIXEXT1:%.*]] = extractelement <4 x float> [[TMP1]], i32 1
// CHECK-NEXT:    [[VECINIT2:%.*]] = insertelement <4 x float> [[VECINIT]], float [[MATRIXEXT1]], i32 1
// CHECK-NEXT:    [[TMP2:%.*]] = load <4 x float>, ptr [[M_ADDR]], align 4
// COL-CHECK-NEXT:    [[MATRIXEXT3:%.*]] = extractelement <4 x float> [[TMP2]], i32 1
// ROW-CHECK-NEXT:    [[MATRIXEXT3:%.*]] = extractelement <4 x float> [[TMP2]], i32 2
// CHECK-NEXT:    [[VECINIT4:%.*]] = insertelement <4 x float> [[VECINIT2]], float [[MATRIXEXT3]], i32 2
// CHECK-NEXT:    [[TMP3:%.*]] = load <4 x float>, ptr [[M_ADDR]], align 4
// CHECK-NEXT:    [[MATRIXEXT5:%.*]] = extractelement <4 x float> [[TMP3]], i32 3
// CHECK-NEXT:    [[VECINIT6:%.*]] = insertelement <4 x float> [[VECINIT4]], float [[MATRIXEXT5]], i32 3
// CHECK-NEXT:    store <4 x float> [[VECINIT6]], ptr [[V]], align 16
// CHECK-NEXT:    [[TMP4:%.*]] = load <4 x float>, ptr [[V]], align 16
// CHECK-NEXT:    ret <4 x float> [[TMP4]]
//
float4 fn(float2x2 m) {
    float4 v = m;
    return v;
}

// CHECK-LABEL: define hidden noundef <4 x i32> @_Z2fnDv4_i(
// CHECK-SAME: <4 x i32> noundef [[V:%.*]]) #[[ATTR0:[0-9]+]] {
// CHECK-NEXT:  [[ENTRY:.*:]]
// CHECK-NEXT:    [[V_ADDR:%.*]] = alloca <4 x i32>, align 16
// CHECK-NEXT:    [[M:%.*]] = alloca [2 x <2 x i32>], align 4
// CHECK-NEXT:    store <4 x i32> [[V]], ptr [[V_ADDR]], align 16
// CHECK-NEXT:    [[TMP0:%.*]] = load <4 x i32>, ptr [[V_ADDR]], align 16
// CHECK-NEXT:    [[VECEXT:%.*]] = extractelement <4 x i32> [[TMP0]], i64 0
// CHECK-NEXT:    [[VECINIT:%.*]] = insertelement <4 x i32> poison, i32 [[VECEXT]], i32 0
// CHECK-NEXT:    [[TMP1:%.*]] = load <4 x i32>, ptr [[V_ADDR]], align 16
// CHECK-NEXT:    [[VECEXT1:%.*]] = extractelement <4 x i32> [[TMP1]], i64 2
// CHECK-NEXT:    [[VECINIT2:%.*]] = insertelement <4 x i32> [[VECINIT]], i32 [[VECEXT1]], i32 1
// CHECK-NEXT:    [[TMP2:%.*]] = load <4 x i32>, ptr [[V_ADDR]], align 16
// CHECK-NEXT:    [[VECEXT3:%.*]] = extractelement <4 x i32> [[TMP2]], i64 1
// CHECK-NEXT:    [[VECINIT4:%.*]] = insertelement <4 x i32> [[VECINIT2]], i32 [[VECEXT3]], i32 2
// CHECK-NEXT:    [[TMP3:%.*]] = load <4 x i32>, ptr [[V_ADDR]], align 16
// CHECK-NEXT:    [[VECEXT5:%.*]] = extractelement <4 x i32> [[TMP3]], i64 3
// CHECK-NEXT:    [[VECINIT6:%.*]] = insertelement <4 x i32> [[VECINIT4]], i32 [[VECEXT5]], i32 3
// CHECK-NEXT:    store <4 x i32> [[VECINIT6]], ptr [[M]], align 4
// CHECK-NEXT:    [[TMP4:%.*]] = load <4 x i32>, ptr [[M]], align 4
// CHECK-NEXT:    ret <4 x i32> [[TMP4]]
//
int2x2 fn(int4 v) {
    int2x2 m = v;
    return m;
}

// CHECK-LABEL: define hidden noundef <2 x i32> @_Z3fn1Dv2_i(
// CHECK-SAME: <2 x i32> noundef [[V:%.*]]) #[[ATTR0]] {
// CHECK-NEXT:  [[ENTRY:.*:]]
// CHECK-NEXT:    [[V_ADDR:%.*]] = alloca <2 x i32>, align 8
// CHECK-NEXT:    store <2 x i32> [[V]], ptr [[V_ADDR]], align 8
// CHECK-NEXT:    [[TMP0:%.*]] = load <2 x i32>, ptr [[V_ADDR]], align 8
// CHECK-NEXT:    [[VECEXT:%.*]] = extractelement <2 x i32> [[TMP0]], i64 0
// CHECK-NEXT:    [[VECINIT:%.*]] = insertelement <2 x i32> poison, i32 [[VECEXT]], i32 0
// CHECK-NEXT:    [[TMP1:%.*]] = load <2 x i32>, ptr [[V_ADDR]], align 8
// CHECK-NEXT:    [[VECEXT1:%.*]] = extractelement <2 x i32> [[TMP1]], i64 1
// CHECK-NEXT:    [[VECINIT2:%.*]] = insertelement <2 x i32> [[VECINIT]], i32 [[VECEXT1]], i32 1
// CHECK-NEXT:    ret <2 x i32> [[VECINIT2]]
//
int1x2 fn1(int2 v) {
    return v;
}

// CHECK-LABEL: define hidden noundef <3 x i1> @_Z3fn2Dv3_b(
// CHECK-SAME: <3 x i1> noundef [[B:%.*]]) #[[ATTR0]] {
// CHECK-NEXT:  [[ENTRY:.*:]]
// CHECK-NEXT:    [[B_ADDR:%.*]] = alloca <3 x i32>, align 16
// CHECK-NEXT:    [[TMP0:%.*]] = zext <3 x i1> [[B]] to <3 x i32>
// CHECK-NEXT:    store <3 x i32> [[TMP0]], ptr [[B_ADDR]], align 16
// CHECK-NEXT:    [[TMP1:%.*]] = load <3 x i32>, ptr [[B_ADDR]], align 16
// CHECK-NEXT:    [[LOADEDV:%.*]] = trunc <3 x i32> [[TMP1]] to <3 x i1>
// CHECK-NEXT:    [[VECEXT:%.*]] = extractelement <3 x i1> [[LOADEDV]], i64 0
// CHECK-NEXT:    [[VECINIT:%.*]] = insertelement <3 x i1> poison, i1 [[VECEXT]], i32 0
// CHECK-NEXT:    [[TMP2:%.*]] = load <3 x i32>, ptr [[B_ADDR]], align 16
// CHECK-NEXT:    [[LOADEDV1:%.*]] = trunc <3 x i32> [[TMP2]] to <3 x i1>
// CHECK-NEXT:    [[VECEXT2:%.*]] = extractelement <3 x i1> [[LOADEDV1]], i64 1
// CHECK-NEXT:    [[VECINIT3:%.*]] = insertelement <3 x i1> [[VECINIT]], i1 [[VECEXT2]], i32 1
// CHECK-NEXT:    [[TMP3:%.*]] = load <3 x i32>, ptr [[B_ADDR]], align 16
// CHECK-NEXT:    [[LOADEDV4:%.*]] = trunc <3 x i32> [[TMP3]] to <3 x i1>
// CHECK-NEXT:    [[VECEXT5:%.*]] = extractelement <3 x i1> [[LOADEDV4]], i64 2
// CHECK-NEXT:    [[VECINIT6:%.*]] = insertelement <3 x i1> [[VECINIT3]], i1 [[VECEXT5]], i32 2
// CHECK-NEXT:    ret <3 x i1> [[VECINIT6]]
//
bool3x1 fn2(bool3 b) {
    return b;
}

// CHECK-LABEL: define hidden noundef <3 x i32> @_Z3fn3u11matrix_typeILm1ELm3EbE(
// CHECK-SAME: <3 x i1> noundef [[B:%.*]]) #[[ATTR0]] {
// CHECK-NEXT:  [[ENTRY:.*:]]
// COL-CHECK-NEXT:    [[B_ADDR:%.*]] = alloca [3 x <1 x i32>], align 4
// ROW-CHECK-NEXT:    [[B_ADDR:%.*]] = alloca [1 x <3 x i32>], align 4
// CHECK-NEXT:    [[TMP0:%.*]] = zext <3 x i1> [[B]] to <3 x i32>
// CHECK-NEXT:    store <3 x i32> [[TMP0]], ptr [[B_ADDR]], align 4
// CHECK-NEXT:    [[TMP1:%.*]] = load <3 x i32>, ptr [[B_ADDR]], align 4
// CHECK-NEXT:    [[MATRIXEXT:%.*]] = extractelement <3 x i32> [[TMP1]], i32 0
// CHECK-NEXT:    [[VECINIT:%.*]] = insertelement <3 x i32> poison, i32 [[MATRIXEXT]], i32 0
// CHECK-NEXT:    [[TMP2:%.*]] = load <3 x i32>, ptr [[B_ADDR]], align 4
// CHECK-NEXT:    [[MATRIXEXT1:%.*]] = extractelement <3 x i32> [[TMP2]], i32 1
// CHECK-NEXT:    [[VECINIT2:%.*]] = insertelement <3 x i32> [[VECINIT]], i32 [[MATRIXEXT1]], i32 1
// CHECK-NEXT:    [[TMP3:%.*]] = load <3 x i32>, ptr [[B_ADDR]], align 4
// CHECK-NEXT:    [[MATRIXEXT3:%.*]] = extractelement <3 x i32> [[TMP3]], i32 2
// CHECK-NEXT:    [[VECINIT4:%.*]] = insertelement <3 x i32> [[VECINIT2]], i32 [[MATRIXEXT3]], i32 2
// CHECK-NEXT:    ret <3 x i32> [[VECINIT4]]
//
int3 fn3(bool1x3 b) {
    return b;
}
