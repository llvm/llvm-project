// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --enable-var-scope

// CHECK-LABEL: define void {{.*}}arr_assign1
// CHECK: [[Arr:%.*]] = alloca [2 x i32], align 4
// CHECK: [[Arr2:%.*]] = alloca [2 x i32], align 4
// CHECK: [[Tmp:%.*]] = alloca [2 x i32], align 4
// CHECK: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[Arr]], ptr align 4 {{@.*}}, i32 8, i1 false)
// CHECK: call void @llvm.memset.p0.i32(ptr align 4 [[Arr2]], i8 0, i32 8, i1 false)
// CHECK: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[Arr]], ptr align 4 [[Arr2]], i32 8, i1 false)
// CHECK: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[Tmp]], ptr align 4 [[Arr]], i32 8, i1 false)
// CHECK: ret void
void arr_assign1() {
  int Arr[2] = {0, 1};
  int Arr2[2] = {0, 0};
  Arr = Arr2;
}

// CHECK-LABEL: define void {{.*}}arr_assign2
// CHECK: [[Arr:%.*]] = alloca [2 x i32], align 4
// CHECK: [[Arr2:%.*]] = alloca [2 x i32], align 4
// CHECK: [[Arr3:%.*]] = alloca [2 x i32], align 4
// CHECK: [[Tmp:%.*]] = alloca [2 x i32], align 4
// CHECK: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[Arr]], ptr align 4 {{@.*}}, i32 8, i1 false)
// CHECK: call void @llvm.memset.p0.i32(ptr align 4 [[Arr2]], i8 0, i32 8, i1 false)
// CHECK: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[Arr3]], ptr align 4 {{@.*}}, i32 8, i1 false)
// CHECK: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[Arr2]], ptr align 4 [[Arr3]], i32 8, i1 false)
// CHECK: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[Arr]], ptr align 4 [[Arr2]], i32 8, i1 false)
// CHECK: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[Tmp]], ptr align 4 [[Arr]], i32 8, i1 false)
// CHECK: ret void
void arr_assign2() {
  int Arr[2] = {0, 1};
  int Arr2[2] = {0, 0};
  int Arr3[2] = {3, 4};
  Arr = Arr2 = Arr3;
}

// CHECK-LABEL: define void {{.*}}arr_assign3
// CHECK: [[Arr3:%.*]] = alloca [2 x [2 x i32]], align 4
// CHECK: [[Arr4:%.*]] = alloca [2 x [2 x i32]], align 4
// CHECK: [[Tmp:%.*]] = alloca [2 x [2 x i32]], align 4
// CHECK: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[Arr3]], ptr align 4 {{@.*}}, i32 16, i1 false)
// CHECK: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[Arr4]], ptr align 4 {{@.*}}, i32 16, i1 false)
// CHECK: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[Arr3]], ptr align 4 [[Arr4]], i32 16, i1 false)
// CHECK: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[Tmp]], ptr align 4 [[Arr3]], i32 16, i1 false)
// CHECK: ret void
void arr_assign3() {
  int Arr2[2][2] = {{0, 0}, {1, 1}};
  int Arr3[2][2] = {{1, 1}, {0, 0}};
  Arr2 = Arr3;
}
