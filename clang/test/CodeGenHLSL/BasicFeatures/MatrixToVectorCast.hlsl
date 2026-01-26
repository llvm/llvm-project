// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.3-library -x hlsl -emit-llvm -disable-llvm-passes -fmatrix-memory-layout=column-major -o - %s | FileCheck %s --check-prefixes=CHECK,COL-CHECK
// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.3-library -x hlsl -emit-llvm -disable-llvm-passes -fmatrix-memory-layout=row-major -o - %s | FileCheck %s --check-prefixes=CHECK,ROW-CHECK

// CHECK-LABEL: define hidden noundef nofpclass(nan inf) <4 x float> @_Z2fnu11matrix_typeILm2ELm2EfE(
// CHECK-SAME: <4 x float> noundef nofpclass(nan inf) [[M:%.*]]) #[[ATTR0:[0-9]+]] {
// CHECK-NEXT:  [[ENTRY:.*:]]
// CHECK-NEXT:    [[M_ADDR:%.*]] = alloca [4 x float], align 4
// CHECK-NEXT:    [[V:%.*]] = alloca <4 x float>, align 16
// CHECK-NEXT:    [[HLSL_EWCAST_SRC:%.*]] = alloca [4 x float], align 4
// CHECK-NEXT:    [[FLATCAST_TMP:%.*]] = alloca <4 x float>, align 16
// CHECK-NEXT:    store <4 x float> [[M]], ptr [[M_ADDR]], align 4
// CHECK-NEXT:    [[TMP0:%.*]] = load <4 x float>, ptr [[M_ADDR]], align 4
// CHECK-NEXT:    store <4 x float> [[TMP0]], ptr [[HLSL_EWCAST_SRC]], align 4
// CHECK-NEXT:    [[MATRIX_GEP:%.*]] = getelementptr inbounds <4 x float>, ptr [[HLSL_EWCAST_SRC]], i32 0
// CHECK-NEXT:    [[TMP1:%.*]] = load <4 x float>, ptr [[FLATCAST_TMP]], align 16
// CHECK-NEXT:    [[TMP2:%.*]] = load <4 x float>, ptr [[MATRIX_GEP]], align 4
// CHECK-NEXT:    [[MATRIXEXT:%.*]] = extractelement <4 x float> [[TMP2]], i32 0
// CHECK-NEXT:    [[TMP3:%.*]] = insertelement <4 x float> [[TMP1]], float [[MATRIXEXT]], i64 0
// CHECK-NEXT:    [[TMP4:%.*]] = load <4 x float>, ptr [[MATRIX_GEP]], align 4
// COL-CHECK-NEXT:    [[MATRIXEXT1:%.*]] = extractelement <4 x float> [[TMP4]], i32 2
// ROW-CHECK-NEXT:    [[MATRIXEXT1:%.*]] = extractelement <4 x float> [[TMP4]], i32 1
// CHECK-NEXT:    [[TMP5:%.*]] = insertelement <4 x float> [[TMP3]], float [[MATRIXEXT1]], i64 1
// CHECK-NEXT:    [[TMP6:%.*]] = load <4 x float>, ptr [[MATRIX_GEP]], align 4
// COL-CHECK-NEXT:    [[MATRIXEXT2:%.*]] = extractelement <4 x float> [[TMP6]], i32 1
// ROW-CHECK-NEXT:    [[MATRIXEXT2:%.*]] = extractelement <4 x float> [[TMP6]], i32 2
// CHECK-NEXT:    [[TMP7:%.*]] = insertelement <4 x float> [[TMP5]], float [[MATRIXEXT2]], i64 2
// CHECK-NEXT:    [[TMP8:%.*]] = load <4 x float>, ptr [[MATRIX_GEP]], align 4
// CHECK-NEXT:    [[MATRIXEXT3:%.*]] = extractelement <4 x float> [[TMP8]], i32 3
// CHECK-NEXT:    [[TMP9:%.*]] = insertelement <4 x float> [[TMP7]], float [[MATRIXEXT3]], i64 3
// CHECK-NEXT:    store <4 x float> [[TMP9]], ptr [[V]], align 16
// CHECK-NEXT:    [[TMP10:%.*]] = load <4 x float>, ptr [[V]], align 16
// CHECK-NEXT:    ret <4 x float> [[TMP10]]
//
float4 fn(float2x2 M) {
    float4 V = (float4)M;
    return V;
}

// CHECK-LABEL: define hidden noundef <3 x i32> @_Z3fn2u11matrix_typeILm3ELm1EiE(
// CHECK-SAME: <3 x i32> noundef [[M:%.*]]) #[[ATTR0:[0-9]+]] {
// CHECK-NEXT:  [[ENTRY:.*:]]
// CHECK-NEXT:    [[M_ADDR:%.*]] = alloca [3 x i32], align 4
// CHECK-NEXT:    [[V:%.*]] = alloca <3 x i32>, align 16
// CHECK-NEXT:    [[HLSL_EWCAST_SRC:%.*]] = alloca [3 x i32], align 4
// CHECK-NEXT:    [[FLATCAST_TMP:%.*]] = alloca <3 x i32>, align 16
// CHECK-NEXT:    store <3 x i32> [[M]], ptr [[M_ADDR]], align 4
// CHECK-NEXT:    [[TMP0:%.*]] = load <3 x i32>, ptr [[M_ADDR]], align 4
// CHECK-NEXT:    store <3 x i32> [[TMP0]], ptr [[HLSL_EWCAST_SRC]], align 4
// CHECK-NEXT:    [[MATRIX_GEP:%.*]] = getelementptr inbounds <3 x i32>, ptr [[HLSL_EWCAST_SRC]], i32 0
// CHECK-NEXT:    [[TMP1:%.*]] = load <3 x i32>, ptr [[FLATCAST_TMP]], align 16
// CHECK-NEXT:    [[TMP2:%.*]] = load <3 x i32>, ptr [[MATRIX_GEP]], align 4
// CHECK-NEXT:    [[MATRIXEXT:%.*]] = extractelement <3 x i32> [[TMP2]], i32 0
// CHECK-NEXT:    [[TMP3:%.*]] = insertelement <3 x i32> [[TMP1]], i32 [[MATRIXEXT]], i64 0
// CHECK-NEXT:    [[TMP4:%.*]] = load <3 x i32>, ptr [[MATRIX_GEP]], align 4
// CHECK-NEXT:    [[MATRIXEXT1:%.*]] = extractelement <3 x i32> [[TMP4]], i32 1
// CHECK-NEXT:    [[TMP5:%.*]] = insertelement <3 x i32> [[TMP3]], i32 [[MATRIXEXT1]], i64 1
// CHECK-NEXT:    [[TMP6:%.*]] = load <3 x i32>, ptr [[MATRIX_GEP]], align 4
// CHECK-NEXT:    [[MATRIXEXT2:%.*]] = extractelement <3 x i32> [[TMP6]], i32 2
// CHECK-NEXT:    [[TMP7:%.*]] = insertelement <3 x i32> [[TMP5]], i32 [[MATRIXEXT2]], i64 2
// CHECK-NEXT:    store <3 x i32> [[TMP7]], ptr [[V]], align 16
// CHECK-NEXT:    [[TMP8:%.*]] = load <3 x i32>, ptr [[V]], align 16
// CHECK-NEXT:    ret <3 x i32> [[TMP8]]
//
int3 fn2(int3x1 M) {
    int3 V = (int3)M;
    return V;
}

