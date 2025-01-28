// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -disable-llvm-passes -emit-llvm -finclude-default-header -o - %s | FileCheck %s

// CHECK-LABEL: increment
void increment(inout int Arr[2]) {
  for (int I = 0; I < 2; I++)
    Arr[0] += 2;
}

// CHECK-LABEL: arrayCall
// CHECK: [[A:%.*]] = alloca [2 x i32], align 4
// CHECK-NEXT: [[Tmp:%.*]] = alloca [2 x i32], align 4
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[A]], ptr align 4 @{{.*}}, i32 8, i1 false)
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[Tmp]], ptr align 4 [[A]], i32 8, i1 false)
// CHECK-NEXT: call void @{{.*}}increment{{.*}}(ptr noalias noundef byval([2 x i32]) align 4 [[Tmp]]) #3
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[A]], ptr align 4 [[Tmp]], i32 8, i1 false)
// CHECK-NEXT: [[Idx:%.*]] = getelementptr inbounds [2 x i32], ptr [[A]], i32 0, i32 0
// CHECK-NEXT: [[B:%.*]] = load i32, ptr [[Idx]], align 4
// CHECK-NEXT: ret i32 [[B]]
export int arrayCall() {
  int A[2] = { 0, 1 };
  increment(A);
  return A[0];
}

// CHECK-LABEL: fn2
void fn2(out int Arr[2]) {
  Arr[0] += 5;
  Arr[1] += 6;
}

// CHECK-LABEL: arrayCall2
// CHECK: [[A:%.*]] = alloca [2 x i32], align 4
// CHECK-NEXT: [[Tmp:%.*]] = alloca [2 x i32], align 4
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[A]], ptr align 4 @{{.*}}, i32 8, i1 false)
// CHECK-NEXT: call void @{{.*}}fn2{{.*}}(ptr noalias noundef byval([2 x i32]) align 4 [[Tmp]]) #3
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[A]], ptr align 4 [[Tmp]], i32 8, i1 false)
// CHECK-NEXT: [[Idx:%.*]] = getelementptr inbounds [2 x i32], ptr [[A]], i32 0, i32 0
// CHECK-NEXT: [[B:%.*]] = load i32, ptr [[Idx]], align 4
// CHECK-NEXT: ret i32 [[B]]
export int arrayCall2() {
  int A[2] = { 0, 1 };
  fn2(A);
  return A[0];
}

// CHECK-LABEL: nestedCall
void nestedCall(inout int Arr[2], uint index) {
  if (index < 2) {
    Arr[index] += 2;
    nestedCall(Arr, index+1);
  }
}

// CHECK-LABEL: arrayCall3
// CHECK: [[A:%.*]] = alloca [2 x i32], align 4
// CHECK-NEXT: [[Tmp:%.*]] = alloca [2 x i32], align 4
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[A]], ptr align 4 @{{.*}}, i32 8, i1 false)
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[Tmp]], ptr align 4 [[A]], i32 8, i1 false)
// CHECK-NEXT: call void @{{.*}}nestedCall{{.*}}(ptr noalias noundef byval([2 x i32]) align 4 [[Tmp]], i32 noundef 0) #3
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[A]], ptr align 4 [[Tmp]], i32 8, i1 false)
// CHECK-NEXT: [[Idx:%.*]] = getelementptr inbounds [2 x i32], ptr [[A]], i32 0, i32 1
// CHECK-NEXT: [[B:%.*]] = load i32, ptr [[Idx]], align 4
// CHECK-NEXt: ret i32 [[B]]
export int arrayCall3() {
  int A[2] = { 0, 1 };
  nestedCall(A, 0);
  return A[1];
}

// CHECK-LABEL: outerCall
// CHECK: [[Tmp:%.*]] = alloca [2 x i32], align 4
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[Tmp]], ptr align 4 %{{.*}}, i32 8, i1 false)
// CHECK-NEXT: call void {{.*}}increment{{.*}}(ptr noalias noundef byval([2 x i32]) align 4 [[Tmp]]) #3
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 4 {{.*}}, ptr align 4 [[Tmp]], i32 8, i1 false)
// CHECK-NEXT: ret void
void outerCall(inout int Arr[2]) {
  increment(Arr);
}

// CHECK-LABEL: arrayCall4
// CHECK: [[A:%.*]] = alloca [2 x i32], align 4
// CHECK-NEXT: [[Tmp:%.*]] = alloca [2 x i32], align 4
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[A]], ptr align 4 @{{.*}}, i32 8, i1 false)
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[Tmp]], ptr align 4 [[A]], i32 8, i1 false)
// CHECK-NEXT: call void @{{.*}}outerCall{{.*}}(ptr noalias noundef byval([2 x i32]) align 4 [[Tmp]]) #3
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[A]], ptr align 4 [[Tmp]], i32 8, i1 false)
// CHECK-NEXT: [[Idx:%.*]] = getelementptr inbounds [2 x i32], ptr [[A]], i32 0, i32 0
// CHECK-NEXT: [[B:%.*]] = load i32, ptr [[Idx]], align 4
// CHECK-NEXT: ret i32 [[B]]
export int arrayCall4() {
  int A[2] = { 0, 1 };
  outerCall(A);
  return A[0];
}

// CHECK-LABEL: fn3
void fn3(int Arr[2]) {}

// CHECK-LABEL: outerCall2
// CHECK: [[Tmp:%.*]] = alloca [2 x i32], align 4
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[Tmp]], ptr align 4 {{.*}}, i32 8, i1 false)
// CHECK-NEXT: call void {{.*}}fn3{{.*}}(ptr noundef byval([2 x i32]) align 4 [[Tmp]]) #3
// CHECK-NEXT: ret void
void outerCall2(inout int Arr[2]) {
  fn3(Arr);
}

// CHECK-LABEL: arrayCall5
// CHECK: [[A:%.*]] = alloca [2 x i32], align 4
// CHECK-NEXT: [[Tmp:%.*]] = alloca [2 x i32], align 4
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[A]], ptr align 4 @{{.*}}, i32 8, i1 false)
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[Tmp]], ptr align 4 [[A]], i32 8, i1 false)
// CHECK-NEXT: call void @{{.*}}outerCall2{{.*}}(ptr noalias noundef byval([2 x i32]) align 4 [[Tmp]]) #3
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[A]], ptr align 4 [[Tmp]], i32 8, i1 false)
// CHECK-NEXT: [[Idx:%.*]] = getelementptr inbounds [2 x i32], ptr [[A]], i32 0, i32 0
// CHECK-NEXT: [[B:%.*]] = load i32, ptr [[Idx]], align 4
// CHECK-NEXT: ret i32 [[B]]
export int arrayCall5() {
  int A[2] = { 0, 1 };
  outerCall2(A);
  return A[0];
}
