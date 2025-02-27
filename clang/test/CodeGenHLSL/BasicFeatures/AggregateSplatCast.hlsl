// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.3-library -x hlsl -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s

// array splat
// CHECK-LABEL: define void {{.*}}call4
// CHECK: [[B:%.*]] = alloca [2 x i32], align 4
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[B]], ptr align 4 {{.*}}, i32 8, i1 false)
// CHECK-NEXT: [[G1:%.*]] = getelementptr inbounds [2 x i32], ptr [[B]], i32 0, i32 0
// CHECK-NEXT: [[G2:%.*]] = getelementptr inbounds [2 x i32], ptr [[B]], i32 0, i32 1
// CHECK-NEXT: store i32 3, ptr [[G1]], align 4
// CHECK-NEXT: store i32 3, ptr [[G2]], align 4
export void call4() {
  int B[2] = {1,2};
  B = (int[2])3;
}

// splat from vector of length 1
// CHECK-LABEL: define void {{.*}}call8
// CHECK: [[A:%.*]] = alloca <1 x i32>, align 4
// CHECK-NEXT: [[B:%.*]] = alloca [2 x i32], align 4
// CHECK-NEXT: store <1 x i32> splat (i32 1), ptr [[A]], align 4
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[B]], ptr align 4 {{.*}}, i32 8, i1 false)
// CHECK-NEXT: [[L:%.*]] = load <1 x i32>, ptr [[A]], align 4
// CHECK-NEXT: [[VL:%.*]] = extractelement <1 x i32> [[L]], i32 0
// CHECK-NEXT: [[G1:%.*]] = getelementptr inbounds [2 x i32], ptr [[B]], i32 0, i32 0
// CHECK-NEXT: [[G2:%.*]] = getelementptr inbounds [2 x i32], ptr [[B]], i32 0, i32 1
// CHECK-NEXT: store i32 [[VL]], ptr [[G1]], align 4
// CHECK-NEXT: store i32 [[VL]], ptr [[G2]], align 4
export void call8() {
  int1 A = {1};
  int B[2] = {1,2};
  B = (int[2])A;
}

// vector splat from vector of length 1
// CHECK-LABEL: define void {{.*}}call1
// CHECK: [[B:%.*]] = alloca <1 x float>, align 4
// CHECK-NEXT: [[A:%.*]] = alloca <4 x i32>, align 16
// CHECK-NEXT: store <1 x float> splat (float 1.000000e+00), ptr [[B]], align 4
// CHECK-NEXT: [[L:%.*]] = load <1 x float>, ptr [[B]], align 4
// CHECK-NEXT: [[VL:%.*]] = extractelement <1 x float> [[L]], i32 0
// CHECK-NEXT: [[C:%.*]] = fptosi float [[VL]] to i32
// CHECK-NEXT: [[SI:%.*]] = insertelement <4 x i32> poison, i32 [[C]], i64 0
// CHECK-NEXT: [[S:%.*]] = shufflevector <4 x i32> [[SI]], <4 x i32> poison, <4 x i32> zeroinitializer
// CHECK-NEXT: store <4 x i32> [[S]], ptr [[A]], align 16
export void call1() {
  float1 B = {1.0};
  int4 A = (int4)B;
}

struct S {
  int X;
  float Y;
};

// struct splats
// CHECK-LABEL: define void {{.*}}call3
// CHECK: [[A:%.*]] = alloca <1 x i32>, align 4
// CHECK: [[s:%.*]] = alloca %struct.S, align 4
// CHECK-NEXT: store <1 x i32> splat (i32 1), ptr [[A]], align 4
// CHECK-NEXT: [[L:%.*]] = load <1 x i32>, ptr [[A]], align 4
// CHECK-NEXT: [[VL:%.*]] = extractelement <1 x i32> [[L]], i32 0
// CHECK-NEXT: [[G1:%.*]] = getelementptr inbounds %struct.S, ptr [[s]], i32 0, i32 0
// CHECK-NEXT: [[G2:%.*]] = getelementptr inbounds %struct.S, ptr [[s]], i32 0, i32 1
// CHECK-NEXT: store i32 [[VL]], ptr [[G1]], align 4
// CHECK-NEXT: [[C:%.*]] = sitofp i32 [[VL]] to float
// CHECK-NEXT: store float [[C]], ptr [[G2]], align 4
export void call3() {
  int1 A = {1};
  S s = (S)A;
}

// struct splat from vector of length 1
// CHECK-LABEL: define void {{.*}}call5
// CHECK: [[A:%.*]] = alloca <1 x i32>, align 4
// CHECK-NEXT: [[s:%.*]] = alloca %struct.S, align 4
// CHECK-NEXT: store <1 x i32> splat (i32 1), ptr [[A]], align 4
// CHECK-NEXT: [[L:%.*]] = load <1 x i32>, ptr [[A]], align 4
// CHECK-NEXT: [[VL:%.*]] = extractelement <1 x i32> [[L]], i32 0
// CHECK-NEXT: [[G1:%.*]] = getelementptr inbounds %struct.S, ptr [[s]], i32 0, i32 0
// CHECK-NEXT: [[G2:%.*]] = getelementptr inbounds %struct.S, ptr [[s]], i32 0, i32 1
// CHECK-NEXT: store i32 [[VL]], ptr [[G1]], align 4
// CHECK-NEXT: [[C:%.*]] = sitofp i32 [[VL]] to float
// CHECK-NEXT: store float [[C]], ptr [[G2]], align 4
export void call5() {
  int1 A = {1};
  S s = (S)A;
}
