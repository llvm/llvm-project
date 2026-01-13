// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -disable-llvm-passes -emit-llvm -finclude-default-header -o - %s | FileCheck %s

// array truncation to a scalar
// CHECK-LABEL: define void {{.*}}call0
// CHECK: [[A:%.*]] = alloca [2 x i32], align 4
// CHECK-NEXT: [[B:%.*]] = alloca float, align 4
// CHECK-NEXT: [[Tmp:%.*]] = alloca [2 x i32], align 4
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[A]], ptr align 4 {{.*}}, i32 8, i1 false)
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[Tmp]], ptr align 4 [[A]], i32 8, i1 false)
// CHECK-NEXT: [[G1:%.*]] = getelementptr inbounds [2 x i32], ptr [[Tmp]], i32 0, i32 0
// CHECK-NEXT: [[G2:%.*]] = getelementptr inbounds [2 x i32], ptr [[Tmp]], i32 0, i32 1
// CHECK-NEXT: [[L:%.*]] = load i32, ptr [[G1]], align 4
// CHECK-NEXT: [[C:%.*]] = sitofp i32 [[L]] to float
// CHECK-NEXT: store float [[C]], ptr [[B]], align 4
export void call0() {
  int A[2] = {0,1};
  float B = (float)A;
}

// array truncation
// CHECK-LABEL: define void {{.*}}call1
// CHECK: [[A:%.*]] = alloca [2 x i32], align 4
// CHECK-NEXT: [[B:%.*]] = alloca [1 x i32], align 4
// CHECK-NEXT: [[Tmp:%.*]] = alloca [2 x i32], align 4
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[A]], ptr align 4 {{.*}}, i32 8, i1 false)
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[B]], ptr align 4 {{.*}}, i32 4, i1 false)
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[Tmp]], ptr align 4 [[A]], i32 8, i1 false)
// CHECK-NEXT: [[G1:%.*]] = getelementptr inbounds [1 x i32], ptr [[B]], i32 0, i32 0
// CHECK-NEXT: [[G2:%.*]] = getelementptr inbounds [2 x i32], ptr [[Tmp]], i32 0, i32 0
// CHECK-NEXT: [[G3:%.*]] = getelementptr inbounds [2 x i32], ptr [[Tmp]], i32 0, i32 1
// CHECK-NEXT: [[L:%.*]] = load i32, ptr [[G2]], align 4
// CHECK-NEXT: store i32 [[L]], ptr [[G1]], align 4
export void call1() {
  int A[2] = {0,1};
  int B[1] = {4};
  B = (int[1])A;
}

// just a cast
// CHECK-LABEL: define void {{.*}}call2
// CHECK: [[A:%.*]] = alloca [1 x i32], align 4
// CHECK-NEXT: [[B:%.*]] = alloca [1 x float], align 4
// CHECK-NEXT: [[Tmp:%.*]] = alloca [1 x i32], align 4
// CHECK-NEXT: call void @llvm.memset.p0.i32(ptr align 4 [[A]], i8 0, i32 4, i1 false)
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[B]], ptr align 4 {{.*}}, i32 4, i1 false)
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[Tmp]], ptr align 4 [[A]], i32 4, i1 false)
// CHECK-NEXT: [[G1:%.*]] = getelementptr inbounds [1 x float], ptr [[B]], i32 0, i32 0
// CHECK-NEXT: [[G2:%.*]] = getelementptr inbounds [1 x i32], ptr [[Tmp]], i32 0, i32 0
// CHECK-NEXT: [[L:%.*]] = load i32, ptr [[G2]], align 4
// CHECK-NEXT: [[C:%.*]] = sitofp i32 [[L]] to float
// CHECK-NEXT: store float [[C]], ptr [[G1]], align 4
export void call2() {
  int A[1] = {0};
  float B[1] = {1.0};
  B = (float[1])A;
}

// vector to array
// CHECK-LABEL: define void {{.*}}call3
// CHECK: [[A:%.*]] = alloca <1 x float>, align 4
// CHECK-NEXT: [[B:%.*]] = alloca [1 x i32], align 4
// CHECK-NEXT: store <1 x float> splat (float 0x3FF3333340000000), ptr [[A]], align 4
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[B]], ptr align 4 {{.*}}, i32 4, i1 false)
// CHECK-NEXT: [[C:%.*]] = load <1 x float>, ptr [[A]], align 4
// CHECK-NEXT: [[G1:%.*]] = getelementptr inbounds [1 x i32], ptr [[B]], i32 0, i32 0
// CHECK-NEXT: [[V:%.*]] = extractelement <1 x float> [[C]], i64 0
// CHECK-NEXT: [[C:%.*]] = fptosi float [[V]] to i32
// CHECK-NEXT: store i32 [[C]], ptr [[G1]], align 4
export void call3() {
  float1 A = {1.2};
  int B[1] = {1};
  B = (int[1])A;
}

// flatten array of vector to array with cast
// CHECK-LABEL: define void {{.*}}call5
// CHECK: [[A:%.*]] = alloca [1 x <2 x float>], align 8
// CHECK-NEXT: [[B:%.*]] = alloca [2 x i32], align 4
// CHECK-NEXT: [[Tmp:%.*]] = alloca [1 x <2 x float>], align 8
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 8 [[A]], ptr align 8 {{.*}}, i32 8, i1 false)
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[B]], ptr align 4 {{.*}}, i32 8, i1 false)
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 8 [[Tmp]], ptr align 8 [[A]], i32 8, i1 false)
// CHECK-NEXT: [[G1:%.*]] = getelementptr inbounds [2 x i32], ptr [[B]], i32 0, i32 0
// CHECK-NEXT: [[G2:%.*]] = getelementptr inbounds [2 x i32], ptr [[B]], i32 0, i32 1
// CHECK-NEXT: [[VG:%.*]] = getelementptr inbounds [1 x <2 x float>], ptr [[Tmp]], i32 0, i32 0
// CHECK-NEXT: [[L:%.*]] = load <2 x float>, ptr [[VG]], align 8
// CHECK-NEXT: [[VL:%.*]] = extractelement <2 x float> [[L]], i32 0
// CHECK-NEXT: [[C:%.*]] = fptosi float [[VL]] to i32
// CHECK-NEXT: store i32 [[C]], ptr [[G1]], align 4
// CHECK-NEXT: [[L4:%.*]] = load <2 x float>, ptr [[VG]], align 8
// CHECK-NEXT: [[VL5:%.*]] = extractelement <2 x float> [[L4]], i32 1
// CHECK-NEXT: [[C6:%.*]] = fptosi float [[VL5]] to i32
// CHECK-NEXT: store i32 [[C6]], ptr [[G2]], align 4
export void call5() {
  float2 A[1] = {{1.2,3.4}};
  int B[2] = {1,2};
  B = (int[2])A;
}

// flatten 2d array
// CHECK-LABEL: define void {{.*}}call6
// CHECK: [[A:%.*]] = alloca [2 x [1 x i32]], align 4
// CHECK-NEXT: [[B:%.*]] = alloca [2 x i32], align 4
// CHECK-NEXT: [[Tmp:%.*]] = alloca [2 x [1 x i32]], align 4
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[A]], ptr align 4 {{.*}}, i32 8, i1 false)
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[B]], ptr align 4 {{.*}}, i32 8, i1 false)
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[Tmp]], ptr align 4 [[A]], i32 8, i1 false)
// CHECK-NEXT: [[G1:%.*]] = getelementptr inbounds [2 x i32], ptr [[B]], i32 0, i32 0
// CHECK-NEXT: [[G2:%.*]] = getelementptr inbounds [2 x i32], ptr [[B]], i32 0, i32 1
// CHECK-NEXT: [[G3:%.*]] = getelementptr inbounds [2 x [1 x i32]], ptr [[Tmp]], i32 0, i32 0, i32 0
// CHECK-NEXT: [[G4:%.*]] = getelementptr inbounds [2 x [1 x i32]], ptr [[Tmp]], i32 0, i32 1, i32 0
// CHECK-NEXT: [[L:%.*]] = load i32, ptr [[G3]], align 4
// CHECK-NEXT: store i32 [[L]], ptr [[G1]], align 4
// CHECK-NEXT: [[L4:%.*]] = load i32, ptr [[G4]], align 4
// CHECK-NEXT: store i32 [[L4]], ptr [[G2]], align 4
export void call6() {
  int A[2][1] = {{1},{3}};
  int B[2] = {1,2};
  B = (int[2])A;
}

struct S {
  int X;
  float Y;
};

// flatten and truncate from a struct
// CHECK-LABEL: define void {{.*}}call7
// CHECK: [[s:%.*]] = alloca %struct.S, align 1
// CHECK-NEXT: [[A:%.*]] = alloca [1 x i32], align 4
// CHECK-NEXT: [[Tmp:%.*]] = alloca %struct.S, align 1
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 1 [[s]], ptr align 1 {{.*}}, i32 8, i1 false)
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[A]], ptr align 4 {{.*}}, i32 4, i1 false)
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 1 [[Tmp]], ptr align 1 [[s]], i32 8, i1 false)
// CHECK-NEXT: [[G1:%.*]] = getelementptr inbounds [1 x i32], ptr [[A]], i32 0, i32 0
// CHECK-NEXT: [[G2:%.*]] = getelementptr inbounds %struct.S, ptr [[Tmp]], i32 0, i32 0
// CHECK-NEXT: [[G3:%.*]] = getelementptr inbounds %struct.S, ptr [[Tmp]], i32 0, i32 1
// CHECK-NEXT: [[L:%.*]] = load i32, ptr [[G2]], align 4
// CHECK-NEXT: store i32 [[L]], ptr [[G1]], align 4
export void call7() {
  S s = {1, 2.9};
  int A[1] = {1};
  A = (int[1])s;
}

struct BFields {
  double D;
  int E: 15;
  int : 8;
  float F;
};

struct Derived : BFields {
  int G;
};

// flatten from a derived struct with bitfields
// CHECK-LABEL: call8
// CHECK: [[A:%.*]] = alloca [4 x i32], align 4
// CHECK-NEXT: [[Tmp:%.*]] = alloca %struct.Derived, align 1
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 1 [[Tmp]], ptr align 1 %D, i32 19, i1 false)
// CHECK-NEXT: [[Gep:%.*]] = getelementptr inbounds [4 x i32], ptr [[A]], i32 0, i32 0
// CHECK-NEXT: [[Gep1:%.*]] = getelementptr inbounds [4 x i32], ptr [[A]], i32 0, i32 1
// CHECK-NEXT: [[Gep2:%.*]] = getelementptr inbounds [4 x i32], ptr [[A]], i32 0, i32 2
// CHECK-NEXT: [[Gep3:%.*]] = getelementptr inbounds [4 x i32], ptr [[A]], i32 0, i32 3
// CHECK-NEXT: [[Gep4:%.*]] = getelementptr inbounds %struct.Derived, ptr [[Tmp]], i32 0, i32 0
// CHECK-NEXT: [[E:%.*]] = getelementptr inbounds nuw %struct.BFields, ptr [[Gep4]], i32 0, i32 1
// CHECK-NEXT: [[Gep5:%.*]] = getelementptr inbounds %struct.Derived, ptr [[Tmp]], i32 0, i32 0, i32 0
// CHECK-NEXT: [[Gep6:%.*]] = getelementptr inbounds %struct.Derived, ptr [[Tmp]], i32 0, i32 0, i32 2
// CHECK-NEXT: [[Gep7:%.*]] = getelementptr inbounds %struct.Derived, ptr [[Tmp]], i32 0, i32 1
// CHECK-NEXT: [[Z:%.*]] = load double, ptr [[Gep5]], align 8
// CHECK-NEXT: [[C:%.*]] = fptosi double [[Z]] to i32
// CHECK-NEXT: store i32 [[C]], ptr [[Gep]], align 4
// CHECK-NEXT: [[BFL:%.*]] = load i24, ptr [[E]], align 1
// CHECK-NEXT: [[BFShl:%.*]] = shl i24 [[BFL]], 9
// CHECK-NEXT: [[BFAshr:%.*]] = ashr i24 [[BFShl]], 9
// CHECK-NEXT: [[BFC:%.*]] = sext i24 [[BFAshr]] to i32
// CHECK-NEXT: store i32 [[BFC]], ptr [[Gep1]], align 4
// CHECK-NEXT: [[Y:%.*]] = load float, ptr [[Gep6]], align 4
// CHECK-NEXT: [[C8:%.*]] = fptosi float [[Y]] to i32
// CHECK-NEXT: store i32 [[C8]], ptr [[Gep2]], align 4
// CHECK-NEXT: [[X:%.*]] = load i32, ptr [[Gep7]], align 4
// CHECK-NEXT: store i32 [[X]], ptr [[Gep3]], align 4
// CHECK-NEXT: ret void
export void call8(Derived D) {
  int A[4] = (int[4])D;  
}
