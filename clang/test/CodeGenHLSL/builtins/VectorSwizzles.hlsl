// RUN: %clang_cc1 -finclude-default-header -fnative-half-type  \
// RUN:   -triple dxil-pc-shadermodel6.3-library %s -disable-llvm-passes \
// RUN:   -emit-llvm -o - | FileCheck %s

// CHECK-LABEL: Single

// Setup local vars.
// CHECK: [[VecAddr:%.*]] = alloca <3 x i64>, align 32
// CHECK-NEXT: [[AAddr:%.*]] = alloca i64, align 8
// CHECK-NEXT: store <3 x i64> %vec, ptr [[VecAddr]], align 32
// CHECK-NEXT: store i64 %a, ptr [[AAddr]], align 8

// Update single element of the vector.
// CHECK-NEXT: [[A:%.*]] = load i64, ptr [[AAddr]], align 8
// CHECK-NEXT: [[Vy:%.*]] = getelementptr <3 x i64>, ptr [[VecAddr]], i32 0, i32 1
// CHECK-NEXT: store i64 [[A]], ptr [[Vy]], align 8

// Return.
// CHECK-NEXT: [[RetVal:%.*]] = load <3 x i64>, ptr [[VecAddr]], align 32
// CHECK-NEXT: ret <3 x i64> [[RetVal]]
uint64_t3 Single(uint64_t3 vec, uint64_t a){
    vec.y = a;
    return vec;
}

// CHECK-LABEL: Double

// Setup local vars.
// CHECK: [[VecAddr:%.*]] = alloca <3 x float>, align 16
// CHECK-NEXT: [[AAddr:%.*]] = alloca float, align 4
// CHECK-NEXT: [[BAddr:%.*]] = alloca float, align 4
// CHECK-NEXT: store <3 x float> %vec, ptr [[VecAddr]], align 16
// CHECK-NEXT: store float %a, ptr [[AAddr]], align 4
// CHECK-NEXT: store float %b, ptr [[BAddr]], align 4

// Create temporary vector {a, b}.
// CHECK-NEXT: [[A:%.*]] = load float, ptr [[AAddr]], align 4
// CHECK-NEXT: [[TmpVec0:%.*]] = insertelement <2 x float> poison, float [[A]], i32 0
// CHECK-NEXT: [[B:%.*]] = load float, ptr [[BAddr]], align 4
// CHECK-NEXT: [[TmpVec1:%.*]] = insertelement <2 x float> [[TmpVec0]], float [[B]], i32 1

// Update two elements of the vector from temporary vector.
// CHECK-NEXT: [[TmpX:%.*]] = extractelement <2 x float> [[TmpVec1]], i32 0
// CHECK-NEXT: [[VecZ:%.*]] = getelementptr <3 x float>, ptr [[VecAddr]], i32 0, i32 2
// CHECK-NEXT: store float [[TmpX]], ptr [[VecZ]], align 4
// CHECK-NEXT: [[TmpY:%.*]] = extractelement <2 x float> [[TmpVec1]], i32 1
// CHECK-NEXT: [[VecY:%.*]] = getelementptr <3 x float>, ptr [[VecAddr]], i32 0, i32 1
// CHECK-NEXT: store float [[TmpY]], ptr [[VecY]], align 4

// Return.
// CHECK-NEXT: [[RetVal:%.*]] = load <3 x float>, ptr [[VecAddr]], align 16
// CHECK-NEXT: ret <3 x float> [[RetVal]]
float3 Double(float3 vec, float a, float b) {
    vec.zy = {a, b};
    return vec;
}

// CHECK-LABEL: Shuffle

// Setup local vars.
// CHECK: [[VecAddr:%.*]] = alloca <4 x half>, align 8
// CHECK-NEXT: [[AAddr:%.*]] = alloca half, align 2
// CHECK-NEXT: [[BAddr:%.*]] = alloca half, align 2
// CHECK-NEXT: store <4 x half> %vec, ptr [[VecAddr]], align 8
// CHECK-NEXT: store half %a, ptr [[AAddr]], align 2
// CHECK-NEXT: store half %b, ptr [[BAddr]], align 2

// Create temporary vector {a, b, 13.74, a}.
// CHECK-NEXT: [[A:%.*]] = load half, ptr [[AAddr]], align 2
// CHECK-NEXT: [[TmpVec0:%.*]] = insertelement <4 x half> poison, half [[A]], i32 0
// CHECK-NEXT: [[B:%.*]] = load half, ptr [[BAddr]], align 2
// CHECK-NEXT: [[TmpVec1:%.*]] = insertelement <4 x half> [[TmpVec0]], half [[B]], i32 1
// CHECK-NEXT: [[TmpVec2:%.*]] = insertelement <4 x half> %vecinit1, half 0xH4ADF, i32 2
// CHECK-NEXT: [[A:%.*]] = load half, ptr [[AAddr]], align 2
// CHECK-NEXT: [[TmpVec3:%.*]] = insertelement <4 x half> [[TmpVec2]], half [[A]], i32 3

// Update four elements of the vector via mixed up swizzle from the temporary vector.
// CHECK-NEXT: [[TmpX:%.*]] = extractelement <4 x half> [[TmpVec3]], i32 0
// CHECK-NEXT: [[VecZ:%.*]] = getelementptr <4 x half>, ptr [[VecAddr]], i32 0, i32 2
// CHECK-NEXT: store half [[TmpX]], ptr [[VecZ]], align 2
// CHECK-NEXT: [[TmpY:%.*]] = extractelement <4 x half> [[TmpVec3]], i32 1
// CHECK-NEXT: [[VecW:%.*]] = getelementptr <4 x half>, ptr [[VecAddr]], i32 0, i32 3
// CHECK-NEXT: store half [[TmpY]], ptr [[VecW]], align 2
// CHECK-NEXT: [[TmpZ:%.*]] = extractelement <4 x half> [[TmpVec3]], i32 2
// CHECK-NEXT: store half [[TmpZ]], ptr [[VecAddr]], align 2
// CHECK-NEXT: [[TmpW:%.*]] = extractelement <4 x half> [[TmpVec3]], i32 3
// CHECK-NEXT: [[VecY:%.*]] = getelementptr <4 x half>, ptr [[VecAddr]], i32 0, i32 1
// CHECK-NEXT: store half [[TmpW]], ptr [[VecY]], align 2

// Return.
// CHECK-NEXT: [[RetVal:%.*]] = load <4 x half>, ptr [[VecAddr]], align 8
// CHECK-NEXT: ret <4 x half> [[RetVal]]
half4 Shuffle(half4 vec, half a, half b) {
    vec.zwxy = {a, b, 13.74, a};
    return vec;
}
