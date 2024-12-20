// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.3-library -x hlsl -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s

struct S {
  int X;
  float Y;
};

// struct truncation to a scalar
// CHECK-LABEL: define void {{.*}}call0
// CHECK: [[s:%.*]] = alloca %struct.S, align 4
// CHECK-NEXT: [[A:%.*]] = alloca i32, align 4
// CHECK-NEXT: [[Tmp:%.*]] = alloca %struct.S, align 4
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[s]], ptr align 4 {{.*}}, i32 8, i1 false)
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[Tmp]], ptr align 4 [[s]], i32 8, i1 false)
// CHECK-NEXT: [[G1:%.*]] = getelementptr inbounds %struct.S, ptr [[Tmp]], i32 0, i32 0
// CHECK-NEXT: [[G2:%.*]] = getelementptr inbounds %struct.S, ptr [[Tmp]], i32 0, i32 1
// CHECK-NEXT: [[L:%.*]] = load i32, ptr [[G1]], align 4
// CHECK-NEXT: store i32 [[L]], ptr [[A]], align 4
export void call0() {
  S s = {1,2};
  int A = (int)s;
}

// struct from vector
// CHECK-LABEL: define void {{.*}}call1
// CHECK: [[A:%.*]] = alloca <2 x i32>, align 8
// CHECK-NEXT: [[s:%.*]] = alloca %struct.S, align 4
// CHECK-NEXT: store <2 x i32> <i32 1, i32 2>, ptr [[A]], align 8
// CHECK-NEXT: [[L:%.*]] = load <2 x i32>, ptr [[A]], align 8
// CHECK-NEXT: [[G1:%.*]] = getelementptr inbounds %struct.S, ptr [[s]], i32 0, i32 0
// CHECK-NEXT: [[G2:%.*]] = getelementptr inbounds %struct.S, ptr [[s]], i32 0, i32 1
// CHECK-NEXT: [[VL:%.*]] = extractelement <2 x i32> [[L]], i64 0
// CHECK-NEXT: store i32 [[VL]], ptr [[G1]], align 4
// CHECK-NEXT: [[VL2:%.*]] = extractelement <2 x i32> [[L]], i64 1
// CHECK-NEXT: [[C:%.*]] = sitofp i32 [[VL2]] to float
// CHECK-NEXT: store float [[C]], ptr [[G2]], align 4
export void call1() {
  int2 A = {1,2};
  S s = (S)A;
}


// struct from array
// CHECK-LABEL: define void {{.*}}call2
// CHECK: [[A:%.*]] = alloca [2 x i32], align 4
// CHECK-NEXT: [[s:%.*]] = alloca %struct.S, align 4
// CHECK-NEXT: [[Tmp:%.*]] = alloca [2 x i32], align 4
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[A]], ptr align 4 {{.*}}, i32 8, i1 false)
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[Tmp]], ptr align 4 [[A]], i32 8, i1 false)
// CHECK-NEXT: [[G1:%.*]] = getelementptr inbounds %struct.S, ptr [[s]], i32 0, i32 0
// CHECK-NEXT: [[G2:%.*]] = getelementptr inbounds %struct.S, ptr [[s]], i32 0, i32 1
// CHECK-NEXT: [[G3:%.*]] = getelementptr inbounds [2 x i32], ptr [[Tmp]], i32 0, i32 0
// CHECK-NEXT: [[G4:%.*]] = getelementptr inbounds [2 x i32], ptr [[Tmp]], i32 0, i32 1
// CHECK-NEXT: [[L:%.*]] = load i32, ptr [[G3]], align 4
// CHECK-NEXT: store i32 [[L]], ptr [[G1]], align 4
// CHECK-NEXT: [[L4:%.*]] = load i32, ptr [[G4]], align 4
// CHECK-NEXT: [[C:%.*]] = sitofp i32 [[L4]] to float
// CHECK-NEXT: store float [[C]], ptr [[G2]], align 4
export void call2() {
  int A[2] = {1,2};
  S s = (S)A;
}

struct Q {
  int Z;
};

struct R {
  Q q;
  float F;
};

// struct from nested struct?
// CHECK-LABEL: define void {{.*}}call6
// CHECK: [[r:%.*]] = alloca %struct.R, align 4
// CHECK-NEXT: [[s:%.*]] = alloca %struct.S, align 4
// CHECK-NEXT: [[Tmp:%.*]] = alloca %struct.R, align 4
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[r]], ptr align 4 {{.*}}, i32 8, i1 false)
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[Tmp]], ptr align 4 [[r]], i32 8, i1 false)
// CHECK-NEXT: [[G1:%.*]] = getelementptr inbounds %struct.S, ptr [[s]], i32 0, i32 0
// CHECK-NEXT: [[G2:%.*]] = getelementptr inbounds %struct.S, ptr [[s]], i32 0, i32 1
// CHECK-NEXT: [[G3:%.*]] = getelementptr inbounds %struct.R, ptr [[Tmp]], i32 0, i32 0
// CHECK-NEXT: [[G4:%.*]] = getelementptr inbounds %struct.R, ptr [[Tmp]], i32 0, i32 1
// CHECK-NEXT: [[L:%.*]] = load i32, ptr [[G3]], align 4
// CHECK-NEXT: store i32 [[L]], ptr [[G1]], align 4
// CHECK-NEXT: [[L4:%.*]] = load float, ptr [[G4]], align 4
// CHECK-NEXT: store float [[L4]], ptr [[G2]], align 4
export void call6() {
  R r = {{1}, 2.0};
  S s = (S)r;
}

// nested struct from array?
// CHECK-LABEL: define void {{.*}}call7
// CHECK: [[A:%.*]] = alloca [2 x i32], align 4
// CHECK-NEXT: [[r:%.*]] = alloca %struct.R, align 4
// CHECK-NEXT: [[Tmp:%.*]] = alloca [2 x i32], align 4
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[A]], ptr align 4 {{.*}}, i32 8, i1 false)
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[Tmp]], ptr align 4 [[A]], i32 8, i1 false)
// CHECK-NEXT: [[G1:%.*]] = getelementptr inbounds %struct.R, ptr [[r]], i32 0, i32 0
// CHECK-NEXT: [[G2:%.*]] = getelementptr inbounds %struct.R, ptr [[r]], i32 0, i32 1
// CHECK-NEXT: [[G3:%.*]] = getelementptr inbounds [2 x i32], ptr [[Tmp]], i32 0, i32 0
// CHECK-NEXT: [[G4:%.*]] = getelementptr inbounds [2 x i32], ptr [[Tmp]], i32 0, i32 1
// CHECK-NEXT: [[L:%.*]] = load i32, ptr [[G3]], align 4
// CHECK-NEXT: store i32 [[L]], ptr [[G1]], align 4
// CHECK-NEXT: [[L4:%.*]] = load i32, ptr [[G4]], align 4
// CHECK-NEXT: [[C:%.*]] = sitofp i32 [[L4]] to float
// CHECK-NEXT: store float [[C]], ptr [[G2]], align 4
export void call7() {
  int A[2] = {1,2};
  R r = (R)A;
}

struct T {
  int A;
  int B;
  int C;
};

// struct truncation
// CHECK-LABEL: define void {{.*}}call8
// CHECK: [[t:%.*]] = alloca %struct.T, align 4
// CHECK-NEXT: [[s:%.*]] = alloca %struct.S, align 4
// CHECK-NEXT: [[Tmp:%.*]] = alloca %struct.T, align 4
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[t]], ptr align 4 {{.*}}, i32 12, i1 false)
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[Tmp]], ptr align 4 [[t]], i32 12, i1 false)
// CHECK-NEXT: [[G1:%.*]] = getelementptr inbounds %struct.S, ptr [[s]], i32 0, i32 0
// CHECK-NEXT: [[G2:%.*]] = getelementptr inbounds %struct.S, ptr [[s]], i32 0, i32 1
// CHECK-NEXT: [[G3:%.*]] = getelementptr inbounds %struct.T, ptr [[Tmp]], i32 0, i32 0
// CHECK-NEXT: %gep3 = getelementptr inbounds %struct.T, ptr %agg-temp, i32 0, i32 1
// CHECK-NEXT: %gep4 = getelementptr inbounds %struct.T, ptr %agg-temp, i32 0, i32 2
// CHECK-NEXT: %load = load i32, ptr %gep2, align 4
// CHECK-NEXT: store i32 %load, ptr %gep, align 4
// CHECK-NEXT: %load5 = load i32, ptr %gep3, align 4
// CHECK-NEXT: %conv = sitofp i32 %load5 to float
// CHECK-NEXT: store float %conv, ptr %gep1, align 4
export void call8() {
  T t = {1,2,3};
  S s = (S)t;
}
