// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.3-library -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s

// CHECK: %struct.S = type { <2 x i32>, float }
// CHECK: [[ConstS:@.*]] = private unnamed_addr constant %struct.S { <2 x i32> splat (i32 1), float 1.000000e+00 }, align 8
struct S {
    bool2 bv;
    float f;
};

// CHECK-LABEL: define noundef i1 {{.*}}fn1{{.*}}
// CHECK: [[B:%.*]] = alloca <2 x i32>, align 8
// CHECK-NEXT: store <2 x i32> splat (i32 1), ptr [[B]], align 8
// CHECK-NEXT: [[BoolVec:%.*]] = load <2 x i32>, ptr [[B]], align 8
// CHECK-NEXT: [[L:%.*]] = trunc <2 x i32> [[BoolVec:%.*]] to <2 x i1>
// CHECK-NEXT: [[VecExt:%.*]] = extractelement <2 x i1> [[L]], i32 0
// CHECK-NEXT: ret i1 [[VecExt]]
bool fn1() {
  bool2 B = {true,true};
  return B[0];
}

// CHECK-LABEL: define noundef <2 x i1> {{.*}}fn2{{.*}}
// CHECK: [[VAddr:%.*]] = alloca i32, align 4
// CHECK-NEXT: [[A:%.*]] = alloca <2 x i32>, align 8
// CHECK-NEXT: [[StoreV:%.*]] = zext i1 {{.*}} to i32
// CHECK-NEXT: store i32 [[StoreV]], ptr [[VAddr]], align 4
// CHECK-NEXT: [[L:%.*]] = load i32, ptr [[VAddr]], align 4
// CHECK-NEXT: [[LoadV:%.*]] = trunc i32 [[L]] to i1
// CHECK-NEXT: [[Vec:%.*]] = insertelement <2 x i1> poison, i1 [[LoadV]], i32 0
// CHECK-NEXT: [[Vec1:%.*]] = insertelement <2 x i1> [[Vec]], i1 true, i32 1
// CHECK-NEXT: [[Z:%.*]] = zext <2 x i1> [[Vec1]] to <2 x i32>
// CHECK-NEXT: store <2 x i32> [[Z]], ptr [[A]], align 8
// CHECK-NEXT: [[LoadBV:%.*]] = load <2 x i32>, ptr [[A]], align 8
// CHECK-NEXT: [[LoadV2:%.*]] = trunc <2 x i32> [[LoadBV]] to <2 x i1>
// CHECK-NEXT: ret <2 x i1> [[LoadV2]]
bool2 fn2(bool V) {
  bool2 A = {V,true};
  return A;
}

// CHECK-LABEL: define noundef i1 {{.*}}fn3{{.*}}
// CHECK: [[s:%.*]] = alloca %struct.S, align 8
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 8 [[s]], ptr align 8 [[ConstS]], i32 16, i1 false)
// CHECK-NEXT: [[BV:%.*]] = getelementptr inbounds nuw %struct.S, ptr [[s]], i32 0, i32 0
// CHECK-NEXT: [[LBV:%.*]] = load <2 x i32>, ptr [[BV]], align 8
// CHECK-NEXT: [[LV:%.*]] = trunc <2 x i32> [[LBV]] to <2 x i1>
// CHECK-NEXT: [[VX:%.*]] = extractelement <2 x i1> [[LV]], i32 0
// CHECK-NEXT: ret i1 [[VX]]
bool fn3() {
  S s = {{true,true}, 1.0};
  return s.bv[0];
}
