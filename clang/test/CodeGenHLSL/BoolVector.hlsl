// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.3-library -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s

// CHECK: %struct.S = type <{ <2 x i32>, float }>
// CHECK: [[ConstS:@.*]] = private unnamed_addr constant %struct.S <{ <2 x i32> splat (i32 1), float 1.000000e+00 }>, align 1
// CHECK: [[ConstArr:.*]] = private unnamed_addr constant [2 x <2 x i32>] [<2 x i32> splat (i32 1), <2 x i32> zeroinitializer], align 8

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
// CHECK: [[s:%.*]] = alloca %struct.S, align 1
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 1 [[s]], ptr align 1 [[ConstS]], i32 12, i1 false)
// CHECK-NEXT: [[BV:%.*]] = getelementptr inbounds nuw %struct.S, ptr [[s]], i32 0, i32 0
// CHECK-NEXT: [[LBV:%.*]] = load <2 x i32>, ptr [[BV]], align 1
// CHECK-NEXT: [[LV:%.*]] = trunc <2 x i32> [[LBV]] to <2 x i1>
// CHECK-NEXT: [[VX:%.*]] = extractelement <2 x i1> [[LV]], i32 0
// CHECK-NEXT: ret i1 [[VX]]
bool fn3() {
  S s = {{true,true}, 1.0};
  return s.bv[0];
}

// CHECK-LABEL: define noundef i1 {{.*}}fn4{{.*}}
// CHECK: [[Arr:%.*]] = alloca [2 x <2 x i32>], align 8
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 8 [[Arr]], ptr align 8 [[ConstArr]], i32 16, i1 false)
// CHECK-NEXT: [[Idx:%.*]] = getelementptr inbounds [2 x <2 x i32>], ptr [[Arr]], i32 0, i32 0
// CHECK-NEXT: [[L:%.*]] = load <2 x i32>, ptr [[Idx]], align 8
// CHECK-NEXT: [[LV:%.*]] = trunc <2 x i32> [[L]] to <2 x i1>
// CHECK-NEXT: [[VX:%.*]] = extractelement <2 x i1> [[LV]], i32 1
// CHECK-NEXT: ret i1 [[VX]]
bool fn4() {
  bool2 Arr[2] = {{true,true}, {false,false}};
  return Arr[0][1];
}

// CHECK-LABEL: define void {{.*}}fn5{{.*}}
// CHECK: [[Arr:%.*]] = alloca <2 x i32>, align 8
// CHECK-NEXT: store <2 x i32> splat (i32 1), ptr [[Arr]], align 8
// CHECK-NEXT: [[L:%.*]] = load <2 x i32>, ptr [[Arr]], align 8
// CHECK-NEXT: [[V:%.*]] = insertelement <2 x i32> [[L]], i32 0, i32 1
// CHECK-NEXT: store <2 x i32> [[V]], ptr [[Arr]], align 8
// CHECK-NEXT: ret void
void fn5() {
  bool2 Arr = {true,true};
  Arr[1] = false;
}

// CHECK-LABEL: define void {{.*}}fn6{{.*}}
// CHECK: [[V:%.*]] = alloca i32, align 4
// CHECK-NEXT: [[S:%.*]] = alloca %struct.S, align 1
// CHECK-NEXT: store i32 0, ptr [[V]], align 4
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 1 [[S]], ptr align 1 {{.*}}, i32 12, i1 false)
// CHECK-NEXT: [[Y:%.*]] = load i32, ptr [[V]], align 4
// CHECK-NEXT: [[LV:%.*]] = trunc i32 [[Y]] to i1
// CHECK-NEXT: [[BV:%.*]] = getelementptr inbounds nuw %struct.S, ptr [[S]], i32 0, i32 0
// CHECK-NEXT: [[X:%.*]] = load <2 x i32>, ptr [[BV]], align 1
// CHECK-NEXT: [[Z:%.*]] = zext i1 [[LV]] to i32
// CHECK-NEXT: [[VI:%.*]] = insertelement <2 x i32> [[X]], i32 [[Z]], i32 1
// CHECK-NEXT: store <2 x i32> [[VI]], ptr [[BV]], align 1
// CHECK-NEXT: ret void
void fn6() {
  bool V = false;
  S s = {{true,true}, 1.0};
  s.bv[1] = V;
}

// CHECK-LABEL: define void {{.*}}fn7{{.*}}
// CHECK: [[Arr:%.*]] = alloca [2 x <2 x i32>], align 8
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 8 [[Arr]], ptr align 8 {{.*}}, i32 16, i1 false)
// CHECK-NEXT: [[Idx:%.*]] = getelementptr inbounds [2 x <2 x i32>], ptr [[Arr]], i32 0, i32 0
// CHECK-NEXT: [[X:%.*]] = load <2 x i32>, ptr [[Idx]], align 8
// CHECK-NEXT: [[VI:%.*]] = insertelement <2 x i32> [[X]], i32 0, i32 1
// CHECK-NEXT: store <2 x i32> [[VI]], ptr [[Idx]], align 8
// CHECK-NEXT: ret void
void fn7() {
  bool2 Arr[2] = {{true,true}, {false,false}};
  Arr[0][1] = false;
}
