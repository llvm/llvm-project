// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.3-library -x hlsl -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s

// vector flat cast from array
// CHECK-LABEL: define void {{.*}}call2
// CHECK: [[A:%.*]] = alloca [2 x [1 x i32]], align 4
// CHECK-NEXT: [[B:%.*]] = alloca <2 x i32>, align 8
// CHECK-NEXT: [[Tmp:%.*]] = alloca [2 x [1 x i32]], align 4
// CHECK-NEXT: [[Tmp2:%.*]] = alloca <2 x i32>, align 8
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[A]], ptr align 4 {{.*}}, i32 8, i1 false)
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[Tmp]], ptr align 4 [[A]], i32 8, i1 false)
// CHECK-NEXT: [[G1:%.*]] = getelementptr inbounds [2 x [1 x i32]], ptr [[Tmp]], i32 0, i32 0, i32 0
// CHECK-NEXT: [[G2:%.*]] = getelementptr inbounds [2 x [1 x i32]], ptr [[Tmp]], i32 0, i32 1, i32 0
// CHECK-NEXT: [[C:%.*]] = load <2 x i32>, ptr [[Tmp2]], align 8
// CHECK-NEXT: [[L:%.*]] = load i32, ptr [[G1]], align 4
// CHECK-NEXT: [[D:%.*]] = insertelement <2 x i32> [[C]], i32 [[L]], i64 0
// CHECK-NEXT: [[L2:%.*]] = load i32, ptr [[G2]], align 4
// CHECK-NEXT: [[E:%.*]] = insertelement <2 x i32> [[D]], i32 [[L2]], i64 1
// CHECK-NEXT: store <2 x i32> [[E]], ptr [[B]], align 8
export void call2() {
  int A[2][1] = {{1},{2}};
  int2 B = (int2)A;
}

struct S {
  int X;
  float Y;
};

// vector flat cast from struct
// CHECK-LABEL: define void {{.*}}call3
// CHECK: [[s:%.*]] = alloca %struct.S, align 1
// CHECK-NEXT: [[A:%.*]] = alloca <2 x i32>, align 8
// CHECK-NEXT: [[Tmp:%.*]] = alloca %struct.S, align 1
// CHECK-NEXT: [[Tmp2:%.*]] = alloca <2 x i32>, align 8
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 1 [[s]], ptr align 1 {{.*}}, i32 8, i1 false)
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 1 [[Tmp]], ptr align 1 [[s]], i32 8, i1 false)
// CHECK-NEXT: [[G1:%.*]] = getelementptr inbounds %struct.S, ptr [[Tmp]], i32 0, i32 0
// CHECK-NEXT: [[G2:%.*]] = getelementptr inbounds %struct.S, ptr [[Tmp]], i32 0, i32 1
// CHECK-NEXT: [[B:%.*]] = load <2 x i32>, ptr [[Tmp2]], align 8
// CHECK-NEXT: [[L:%.*]] = load i32, ptr [[G1]], align 4
// CHECK-NEXT: [[C:%.*]] = insertelement <2 x i32> [[B]], i32 [[L]], i64 0
// CHECK-NEXT: [[L2:%.*]] = load float, ptr [[G2]], align 4
// CHECK-NEXT: [[D:%.*]] = fptosi float [[L2]] to i32
// CHECK-NEXT: [[E:%.*]] = insertelement <2 x i32> [[C]], i32 [[D]], i64 1
// CHECK-NEXT: store <2 x i32> [[E]], ptr [[A]], align 8
export void call3() {
  S s = {1, 2.0};
  int2 A = (int2)s;
}

// truncate array to scalar
// CHECK-LABEL: define void {{.*}}call4
// CHECK: [[A:%.*]] = alloca [2 x i32], align 4
// CHECK-NEXT: [[B:%.*]] = alloca i32, align 4
// CHECK-NEXT: [[Tmp:%.*]] = alloca [2 x i32], align 4
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[A]], ptr align 4 {{.*}}, i32 8, i1 false)
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[Tmp]], ptr align 4 [[A]], i32 8, i1 false)
// CHECK-NEXT: [[G1:%.*]] = getelementptr inbounds [2 x i32], ptr [[Tmp]], i32 0, i32 0
// CHECK-NEXT: [[G2:%.*]] = getelementptr inbounds [2 x i32], ptr [[Tmp]], i32 0, i32 1
// CHECK-NEXT: [[L:%.*]] = load i32, ptr [[G1]], align 4
// CHECK-NEXT: store i32 [[L]], ptr [[B]], align 4
export void call4() {
 int A[2] = {1,2};
 int B = (int)A;
}

// truncate struct to scalar
// CHECK-LABEL: define void {{.*}}call5
// CHECK: [[s:%.*]] = alloca %struct.S, align 1
// CHECK-NEXT: [[A:%.*]] = alloca i32, align 4
// CHECK-NEXT: [[Tmp:%.*]] = alloca %struct.S, align 1
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 1 [[s]], ptr align 1 {{.*}}, i32 8, i1 false)
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 1 [[Tmp]], ptr align 1 [[s]], i32 8, i1 false)
// CHECK-NEXT: [[G1:%.*]] = getelementptr inbounds %struct.S, ptr [[Tmp]], i32 0, i32 0
// CHECK-NEXT: [[G2:%.*]] = getelementptr inbounds %struct.S, ptr [[Tmp]], i32 0, i32 1
// CHECK-NEXT: [[L:%.*]] = load i32, ptr [[G1]], align 4
// CHECK-NEXT: store i32 [[L]], ptr [[A]], align 4
export void call5() {
 S s = {1, 2.0};
 int A = (int)s;
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

// vector flat cast from derived struct with bitfield
// CHECK-LABEL: call6
// CHECK: [[A:%.*]] = alloca <4 x i32>, align 16
// CHECK-NEXT: [[Tmp:%.*]] = alloca %struct.Derived, align 1
// CHECK-NEXT: [[FlatTmp:%.*]] = alloca <4 x i32>, align 16
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 1 [[Tmp]], ptr align 1 %D, i32 19, i1 false)
// CHECK-NEXT: [[Gep:%.*]] = getelementptr inbounds %struct.Derived, ptr [[Tmp]], i32 0, i32 0
// CHECK-NEXT: [[E:%.*]] = getelementptr inbounds nuw %struct.BFields, ptr [[Gep]], i32 0, i32 1
// CHECK-NEXT: [[Gep1:%.*]] = getelementptr inbounds %struct.Derived, ptr [[Tmp]], i32 0, i32 0, i32 0
// CHECK-NEXT: [[Gep2:%.*]] = getelementptr inbounds %struct.Derived, ptr [[Tmp]], i32 0, i32 0, i32 2
// CHECK-NEXT: [[Gep3:%.*]] = getelementptr inbounds %struct.Derived, ptr [[Tmp]], i32 0, i32 1
// CHECK-NEXT: [[Z:%.*]] = load <4 x i32>, ptr [[FlatTmp]], align 16
// CHECK-NEXT: [[Y:%.*]] = load double, ptr [[Gep1]], align 8
// CHECK-NEXT: [[C:%.*]] = fptosi double [[Y]] to i32
// CHECK-NEXT: [[X:%.*]] = insertelement <4 x i32> [[Z]], i32 [[C]], i64 0
// CHECK-NEXT: [[BFL:%.*]] = load i24, ptr [[E]], align 1
// CHECK-NEXT: [[BFShl:%.*]] = shl i24 [[BFL]], 9
// CHECK-NEXT: [[BFAshr:%.*]] = ashr i24 [[BFShl]], 9
// CHECK-NEXT: [[BFC:%.*]] = sext i24 [[BFAshr]] to i32
// CHECK-NEXT: [[W:%.*]] = insertelement <4 x i32> [[X]], i32 [[BFC]], i64 1
// CHECK-NEXT: [[V:%.*]] = load float, ptr [[Gep2]], align 4
// CHECK-NEXT: [[C4:%.*]] = fptosi float [[V]] to i32
// CHECK-NEXT: [[U:%.*]] = insertelement <4 x i32> [[W]], i32 [[C4]], i64 2
// CHECK-NEXT: [[T:%.*]] = load i32, ptr [[Gep3]], align 4
// CHECK-NEXT: [[S:%.*]] = insertelement <4 x i32> [[U]], i32 [[T]], i64 3
// CHECK-NEXT: store <4 x i32> [[S]], ptr [[A]], align 16
// CHECK-NEXT: ret void
export void call6(Derived D) {
  int4 A = (int4)D;
}
