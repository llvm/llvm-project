// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -finclude-default-header -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s

struct Base {
  int A, B;
};

struct Derived: Base {
  float F, G;
};

struct Other {
  int C, D;
};

export void fn() {
// CHECK: [[B:%.*]] = alloca %struct.Base, align 1
// CHECK-NEXT: [[C:%.*]]  = alloca %struct.Base, align 1
// CHECK-NEXT: [[Tmp:%.*]] = alloca %struct.Base, align 1
// CHECK-NEXT: [[O:%.*]] = alloca %struct.Other, align 1
// CHECK-NEXT: [[Tmp1:%.*]] = alloca %struct.Base, align 1
// CHECK-NEXT: [[AggTmp:%.*]] = alloca %struct.Other, align 1
// CHECK-NEXT: [[I2:%.*]] = alloca <2 x i32>, align 4
// CHECK-NEXT: [[Tmp5:%.*]] = alloca %struct.Base, align 1
// CHECK-NEXT: [[D:%.*]] = alloca %struct.Derived, align 1
// CHECK-NEXT: [[Tmp9:%.*]] = alloca %struct.Base, align 1
// CHECK-NEXT: [[AggTmp10:%.*]] = alloca %struct.Derived, align 1

// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 1 [[B]], ptr align 1 @__const._Z2fnv.B, i32 8, i1 false)
  Base B = {1,2};
  
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 1 [[C]], ptr align 1 @__const._Z2fnv.C, i32 8, i1 false)
  Base C = {5,6};

// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 1 [[B]], ptr align 1 [[C]], i32 8, i1 false)
// These Tmp assignments are the "result" of the assignment being stored...
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 1 [[Tmp]], ptr align 1 [[B]], i32 8, i1 false)
  B = C;

// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 1 [[O]], ptr align 1 @__const._Z2fnv.O, i32 8, i1 false)
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 1 [[AggTmp]], ptr align 1 [[O]], i32 8, i1 false)
  Other O = {7,8};

// CHECK-NEXT: [[A1:%.*]] = getelementptr inbounds %struct.Base, ptr [[C]], i32 0, i32 0
// CHECK-NEXT: [[B1:%.*]] = getelementptr inbounds %struct.Base, ptr [[C]], i32 0, i32 1
// CHECK-NEXT: [[A2:%.*]] = getelementptr inbounds %struct.Other, ptr [[AggTmp]], i32 0, i32 0
// CHECK-NEXT: [[B2:%.*]] = getelementptr inbounds %struct.Other, ptr [[AggTmp]], i32 0, i32 1
// CHECK-NEXT: [[A3:%.*]] = load i32, ptr [[A2]], align 4
// CHECK-NEXT: store i32 [[A3]], ptr [[A1]], align 4
// CHECK-NEXT: [[B3:%.*]] = load i32, ptr [[B2]], align 4
// CHECK-NEXT: store i32 [[B3]], ptr [[B1]], align 4
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 1 [[Tmp1]], ptr align 1 [[C]], i32 8, i1 false)
// ^ "result" of the assignment is C so it gets stored to tmp1
  C = (Base)O;

// CHECK-NEXT:  store <2 x i32> <i32 9, i32 10>, ptr %I2, align 4
  int2 I2 = {9,10};

// CHECK-NEXT: [[I3:%.*]] = load <2 x i32>, ptr [[I2]], align 4
// CHECK-NEXT: [[A1:%.*]] = getelementptr inbounds %struct.Base, ptr [[C]], i32 0, i32 0
// CHECK-NEXT: [[B1:%.*]] = getelementptr inbounds %struct.Base, ptr [[C]], i32 0, i32 1
// CHECK-NEXT: [[L0:%.*]] = extractelement <2 x i32> [[I3]], i64 0
// CHECK-NEXT: store i32 [[L0]], ptr [[A1]], align 4
// CHECK-NEXT: [[L1:%.*]] = extractelement <2 x i32> [[I3]], i64 1
// CHECK-NEXT: store i32 [[L1]], ptr [[B1]], align 4
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 1 [[Tmp5]], ptr align 1 [[C]], i32 8, i1 false)
// ^ the result of the assignment is C so it gets stored to tmp5
  C = (Base)I2;


// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 1 [[D]], ptr align 1 @__const._Z2fnv.D, i32 16, i1 false)
  Derived D = {1,2,3,4};

// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 1 [[AggTmp10]], ptr align 1 [[D]], i32 16, i1 false)
// CHECK-NEXT: [[A1:%.*]] = getelementptr inbounds %struct.Base, ptr [[B]], i32 0, i32 0
// CHECK-NEXT: [[B1:%.*]] = getelementptr inbounds %struct.Base, ptr [[B]], i32 0, i32 1
// CHECK-NEXT: [[A2:%.*]] = getelementptr inbounds %struct.Derived, ptr [[AggTmp10]], i32 0, i32 0, i32 0
// CHECK-NEXT: [[B2:%.*]] = getelementptr inbounds %struct.Derived, ptr [[AggTmp10]], i32 0, i32 0, i32 1
// CHECK-NEXT: [[F1:%.*]] = getelementptr inbounds %struct.Derived, ptr [[AggTmp10]], i32 0, i32 1
// CHECK-NEXT: [[G1:%.*]] = getelementptr inbounds %struct.Derived, ptr [[AggTmp10]], i32 0, i32 2
// CHECK-NEXT: [[A3:%.*]] = load i32, ptr [[A2]], align 4
// CHECK-NEXT: store i32 [[A3]], ptr [[A1]], align 4
// CHECK-NEXT: [[B3:%.*]] = load i32, ptr [[B2]], align 4
// CHECK-NEXT: store i32 [[B3]], ptr [[B1]], align 4
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 1 [[Tmp9]], ptr align 1 [[B]], i32 8, i1 false)
// ^ the result of the assignment is B so it gets stored to tmp9
  B = (Base)D;
}
