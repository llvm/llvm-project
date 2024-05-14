// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s

void fn(float x[2]) { }

// CHECK-LABEL: define void {{.*}}call{{.*}}
// CHECK: [[Arr:%.*]] = alloca [2 x float]
// CHECK: [[Tmp:%.*]] = alloca [2 x float]
// CHECK: call void @llvm.memset.p0.i32(ptr align 4 [[Arr]], i8 0, i32 8, i1 false)
// CHECK: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[Tmp]], ptr align 4 [[Arr]], i32 8, i1 false)
// CHECK: call void {{.*}}fn{{.*}}(ptr noundef byval([2 x float]) align 4 [[Tmp]])
void call() {
  float Arr[2] = {0, 0};
  fn(Arr);
}

struct Obj {
  float V;
  int X;
};

void fn2(Obj O[4]) { }

// CHECK-LABEL: define void {{.*}}call2{{.*}}
// CHECK: [[Arr:%.*]] = alloca [4 x %struct.Obj]
// CHECK: [[Tmp:%.*]] = alloca [4 x %struct.Obj]
// CHECK: call void @llvm.memset.p0.i32(ptr align 4 [[Arr]], i8 0, i32 32, i1 false)
// CHECK: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[Tmp]], ptr align 4 [[Arr]], i32 32, i1 false)
// CHECK: call void {{.*}}fn2{{.*}}(ptr noundef byval([4 x %struct.Obj]) align 4 [[Tmp]])
void call2() {
  Obj Arr[4] = {};
  fn2(Arr);
}


void fn3(float x[2][2]) { }

// CHECK-LABEL: define void {{.*}}call3{{.*}}
// CHECK: [[Arr:%.*]] = alloca [2 x [2 x float]]
// CHECK: [[Tmp:%.*]] = alloca [2 x [2 x float]]
// CHECK: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[Arr]], ptr align 4 {{.*}}, i32 16, i1 false)
// CHECK: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[Tmp]], ptr align 4 [[Arr]], i32 16, i1 false)
// CHECK: call void {{.*}}fn3{{.*}}(ptr noundef byval([2 x [2 x float]]) align 4 [[Tmp]])
void call3() {
  float Arr[2][2] = {{0, 0}, {1,1}};
  fn3(Arr);
}

// CHECK-LABEL: define void {{.*}}call4{{.*}}(ptr
// CHECK-SAME: noundef byval([2 x [2 x float]]) align 4 [[Arr:%.*]])
// CHECK: [[Tmp:%.*]] = alloca [2 x [2 x float]]
// CHECK: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[Tmp]], ptr align 4 [[Arr]], i32 16, i1 false)
// CHECK: call void {{.*}}fn3{{.*}}(ptr noundef byval([2 x [2 x float]]) align 4 [[Tmp]])

void call4(float Arr[2][2]) {
  fn3(Arr);
}

// Verify that each template instantiation codegens to a unique and correctly
// mangled function name.

// CHECK-LABEL: define void {{.*}}template_call{{.*}}(ptr

// CHECK-SAME: noundef byval([2 x float]) align 4 [[FA2:%[0-9A-Z]+]],
// CHECK-SAME: ptr noundef byval([4 x float]) align 4 [[FA4:%[0-9A-Z]+]],
// CHECK-SAME: ptr noundef byval([3 x i32]) align 4 [[IA3:%[0-9A-Z]+]]

// CHECK: [[Tmp1:%.*]] = alloca [2 x float]
// CHECK: [[Tmp2:%.*]] = alloca [4 x float]
// CHECK: [[Tmp3:%.*]] = alloca [3 x i32]
// CHECK: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[Tmp1]], ptr align 4 [[FA2]], i32 8, i1 false)
// CHECK: call void @"??$template_fn@$$BY01M@@YAXY01M@Z"(ptr noundef byval([2 x float]) align 4 [[Tmp1]])
// CHECK: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[Tmp2]], ptr align 4 [[FA4]], i32 16, i1 false)
// CHECK: call void @"??$template_fn@$$BY03M@@YAXY03M@Z"(ptr noundef byval([4 x float]) align 4 [[Tmp2]])
// CHECK: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[Tmp3]], ptr align 4 [[IA3]], i32 12, i1 false)
// CHECK: call void @"??$template_fn@$$BY02H@@YAXY02H@Z"(ptr noundef byval([3 x i32]) align 4 [[Tmp3]])

template<typename T>
void template_fn(T Val) {}

void template_call(float FA2[2], float FA4[4], int IA3[3]) {
  template_fn(FA2);
  template_fn(FA4);
  template_fn(IA3);
}


// Verify that Array parameter element access correctly codegens.
// CHECK-LABEL: define void {{.*}}element_access{{.*}}(ptr
// CHECK-SAME: noundef byval([2 x float]) align 4 [[FA2:%[0-9A-Z]+]]

// CHECK: [[Addr:%.*]] = getelementptr inbounds [2 x float], ptr [[FA2]], i32 0, i32 0
// CHECK: [[Tmp:%.*]] = load float, ptr [[Addr]]
// CHECK: call void @"??$template_fn@M@@YAXM@Z"(float noundef [[Tmp]])

// CHECK: [[Idx0:%.*]] = getelementptr inbounds [2 x float], ptr [[FA2]], i32 0, i32 0
// CHECK: [[Val0:%.*]] = load float, ptr [[Idx0]]
// CHECK: [[Sum:%.*]] = fadd float [[Val0]], 5.000000e+00
// CHECK: [[Idx1:%.*]] = getelementptr inbounds [2 x float], ptr [[FA2]], i32 0, i32 1
// CHECK: store float [[Sum]], ptr [[Idx1]]

void element_access(float FA2[2]) {
  template_fn(FA2[0]);
  FA2[1] = FA2[0] + 5;
}
