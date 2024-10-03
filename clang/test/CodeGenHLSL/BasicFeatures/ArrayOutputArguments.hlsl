// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -disable-llvm-passes -emit-llvm -finclude-default-header -o - %s | FileCheck %s

// CHECK-LABEL: increment
void increment(inout int Arr[2]) {
  for (int I = 0; I < 2; I++)
    Arr[0] += 2;
}

// CHECK-LABEL: call
// CHECK: [[A:%.*]] = alloca [2 x i32], align 4
// CHECK-NEXT: [[Tmp:%.*]] = alloca [2 x i32], align 4
// CHECK-NEXT: [[Tmp2:%.*]] = alloca [2 x i32], align 4
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[A]], ptr align 4 @{{.*}}, i32 8, i1 false)
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[Tmp]], ptr align 4 [[A]], i32 8, i1 false)
// CHECK-NEXT: store ptr [[Tmp]], ptr [[Tmp2]], align 4
// CHECK-NEXT: call void @"?increment{{.*}}(ptr noalias noundef byval([2 x i32]) align 4 [[Tmp2]]) #3
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[A]], ptr align 4 [[Tmp]], i32 8, i1 false)
// CHECK-NEXT: [[Idx:%.*]] = getelementptr inbounds [2 x i32], ptr [[A]], i32 0, i32 0
// CHECK-NEXT: [[B:%.*]] = load i32, ptr [[Idx]], align 4
// CHECK-NEXT: ret i32 [[B]]
export int call() {
  int A[2] = { 0, 1 };
  increment(A);
  return A[0];
}

// CHECK-LABEL: fn2
void fn2(out int Arr[2]) {
  Arr[0] += 5;
  Arr[1] += 6;
}

// CHECK-LABEL: call2
// CHECK: [[A:%.*]] = alloca [2 x i32], align 4
// CHECK-NEXT: [[Tmp:%.*]] = alloca [2 x i32], align 4
// CHECK-NEXT: [[Tmp2:%.*]] = alloca [2 x i32], align 4
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[A]], ptr align 4 @{{.*}}, i32 8, i1 false)
// CHECK-NEXT: store ptr [[Tmp]], ptr [[Tmp2]], align 4
// CHECK-NEXT: call void @"?fn2{{.*}}(ptr noalias noundef byval([2 x i32]) align 4 [[Tmp2]]) #3
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[A]], ptr align 4 [[Tmp]], i32 8, i1 false)
// CHECK-NEXT: [[Idx:%.*]] = getelementptr inbounds [2 x i32], ptr [[A]], i32 0, i32 0
// CHECK-NEXT: [[B:%.*]] = load i32, ptr [[Idx]], align 4
// CHECK-NEXT: ret i32 [[B]]
export int call2() {
  int A[2] = { 0, 1 };
  fn2(A);
  return A[0];
}
