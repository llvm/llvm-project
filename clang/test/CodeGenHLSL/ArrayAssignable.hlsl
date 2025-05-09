// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -finclude-default-header -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s

struct S {
  int x;
  float f;
};

// CHECK: [[CBLayout:%.*]] = type <{ [2 x float], [2 x <4 x i32>], [2 x [2 x i32]], [1 x target("dx.Layout", %S, 8, 0, 4)] }>
// CHECK: @CBArrays.cb = global target("dx.CBuffer", target("dx.Layout", [[CBLayout]], 136, 0, 32, 64, 128))
// CHECK: @c1 = external addrspace(2) global [2 x float], align 4
// CHECK: @c2 = external addrspace(2) global [2 x <4 x i32>], align 16
// CHECK: @c3 = external addrspace(2) global [2 x [2 x i32]], align 4
// CHECK: @c4 = external addrspace(2) global [1 x target("dx.Layout", %S, 8, 0, 4)], align 1

cbuffer CBArrays : register(b0) {
  float c1[2];
  int4 c2[2];
  int c3[2][2];
  S c4[1];
}

// CHECK-LABEL: define void {{.*}}arr_assign1
// CHECK: [[Arr:%.*]] = alloca [2 x i32], align 4
// CHECK-NEXT: [[Arr2:%.*]] = alloca [2 x i32], align 4
// CHECK-NOT: alloca
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[Arr]], ptr align 4 {{@.*}}, i32 8, i1 false)
// CHECK-NEXT: call void @llvm.memset.p0.i32(ptr align 4 [[Arr2]], i8 0, i32 8, i1 false)
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[Arr]], ptr align 4 [[Arr2]], i32 8, i1 false)
// CHECK-NEXT: ret void
void arr_assign1() {
  int Arr[2] = {0, 1};
  int Arr2[2] = {0, 0};
  Arr = Arr2;
}

// CHECK-LABEL: define void {{.*}}arr_assign2
// CHECK: [[Arr:%.*]] = alloca [2 x i32], align 4
// CHECK-NEXT: [[Arr2:%.*]] = alloca [2 x i32], align 4
// CHECK-NEXT: [[Arr3:%.*]] = alloca [2 x i32], align 4
// CHECK-NOT: alloca
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[Arr]], ptr align 4 {{@.*}}, i32 8, i1 false)
// CHECK-NEXT: call void @llvm.memset.p0.i32(ptr align 4 [[Arr2]], i8 0, i32 8, i1 false)
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[Arr3]], ptr align 4 {{@.*}}, i32 8, i1 false)
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[Arr2]], ptr align 4 [[Arr3]], i32 8, i1 false)
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[Arr]], ptr align 4 [[Arr2]], i32 8, i1 false)
// CHECK-NEXT: ret void
void arr_assign2() {
  int Arr[2] = {0, 1};
  int Arr2[2] = {0, 0};
  int Arr3[2] = {3, 4};
  Arr = Arr2 = Arr3;
}

// CHECK-LABEL: define void {{.*}}arr_assign3
// CHECK: [[Arr3:%.*]] = alloca [2 x [2 x i32]], align 4
// CHECK-NEXT: [[Arr4:%.*]] = alloca [2 x [2 x i32]], align 4
// CHECK-NOT: alloca
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[Arr3]], ptr align 4 {{@.*}}, i32 16, i1 false)
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[Arr4]], ptr align 4 {{@.*}}, i32 16, i1 false)
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[Arr3]], ptr align 4 [[Arr4]], i32 16, i1 false)
// CHECK-NEXT: ret void
void arr_assign3() {
  int Arr2[2][2] = {{0, 0}, {1, 1}};
  int Arr3[2][2] = {{1, 1}, {0, 0}};
  Arr2 = Arr3;
}

// CHECK-LABEL: define void {{.*}}arr_assign4
// CHECK: [[Arr:%.*]] = alloca [2 x i32], align 4
// CHECK-NEXT: [[Arr2:%.*]] = alloca [2 x i32], align 4
// CHECK-NOT: alloca
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[Arr]], ptr align 4 {{@.*}}, i32 8, i1 false)
// CHECK-NEXT: call void @llvm.memset.p0.i32(ptr align 4 [[Arr2]], i8 0, i32 8, i1 false)
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[Arr]], ptr align 4 [[Arr2]], i32 8, i1 false)
// CHECK-NEXT: [[Idx:%.*]] = getelementptr inbounds [2 x i32], ptr [[Arr]], i32 0, i32 0
// CHECK-NEXT: store i32 6, ptr [[Idx]], align 4
// CHECK-NEXT: ret void
void arr_assign4() {
  int Arr[2] = {0, 1};
  int Arr2[2] = {0, 0};
  (Arr = Arr2)[0] = 6;
}

// CHECK-LABEL: define void {{.*}}arr_assign5
// CHECK: [[Arr:%.*]] = alloca [2 x i32], align 4
// CHECK-NEXT: [[Arr2:%.*]] = alloca [2 x i32], align 4
// CHECK-NEXT: [[Arr3:%.*]] = alloca [2 x i32], align 4
// CHECK-NOT: alloca
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[Arr]], ptr align 4 {{@.*}}, i32 8, i1 false)
// CHECK-NEXT: call void @llvm.memset.p0.i32(ptr align 4 [[Arr2]], i8 0, i32 8, i1 false)
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[Arr3]], ptr align 4 {{@.*}}, i32 8, i1 false)
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[Arr2]], ptr align 4 [[Arr3]], i32 8, i1 false)
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[Arr]], ptr align 4 [[Arr2]], i32 8, i1 false)
// CHECK-NEXT: [[Idx:%.*]] = getelementptr inbounds [2 x i32], ptr [[Arr]], i32 0, i32 0
// CHECK-NEXT: store i32 6, ptr [[Idx]], align 4
// CHECK-NEXT: ret void
void arr_assign5() {
  int Arr[2] = {0, 1};
  int Arr2[2] = {0, 0};
  int Arr3[2] = {3, 4};
  (Arr = Arr2 = Arr3)[0] = 6;
}

// CHECK-LABEL: define void {{.*}}arr_assign6
// CHECK: [[Arr3:%.*]] = alloca [2 x [2 x i32]], align 4
// CHECK-NEXT: [[Arr4:%.*]] = alloca [2 x [2 x i32]], align 4
// CHECK-NOT: alloca
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[Arr3]], ptr align 4 {{@.*}}, i32 16, i1 false)
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[Arr4]], ptr align 4 {{@.*}}, i32 16, i1 false)
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[Arr3]], ptr align 4 [[Arr4]], i32 16, i1 false)
// CHECK-NEXT: [[Idx:%.*]] = getelementptr inbounds [2 x [2 x i32]], ptr [[Arr3]], i32 0, i32 0
// CHECK-NEXT: [[Idx2:%.*]] = getelementptr inbounds [2 x i32], ptr [[Idx]], i32 0, i32 0
// CHECK-NEXT: store i32 6, ptr [[Idx2]], align 4
// CHECK-NEXT: ret void
void arr_assign6() {
  int Arr[2][2] = {{0, 0}, {1, 1}};
  int Arr2[2][2] = {{1, 1}, {0, 0}};
  (Arr = Arr2)[0][0] = 6;
}

// CHECK-LABEL: define void {{.*}}arr_assign7
// CHECK: [[Arr:%.*]] = alloca [2 x [2 x i32]], align 4
// CHECK-NEXT: [[Arr2:%.*]] = alloca [2 x [2 x i32]], align 4
// CHECK-NOT: alloca
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[Arr]], ptr align 4 {{@.*}}, i32 16, i1 false)
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[Arr2]], ptr align 4 {{@.*}}, i32 16, i1 false)
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[Arr]], ptr align 4 [[Arr2]], i32 16, i1 false)
// CHECK-NEXT: [[Idx:%.*]] = getelementptr inbounds [2 x [2 x i32]], ptr [[Arr]], i32 0, i32 0
// CHECK-NEXT: store i32 6, ptr [[Idx]], align 4
// CHECK-NEXT: [[Idx2:%.*]] = getelementptr inbounds i32, ptr %arrayidx, i32 1
// CHECK-NEXT: store i32 6, ptr [[Idx2]], align 4
// CHECK-NEXT: ret void
void arr_assign7() {
  int Arr[2][2] = {{0, 1}, {2, 3}};
  int Arr2[2][2] = {{0, 0}, {1, 1}};
  (Arr = Arr2)[0] = {6, 6};
}

// Verify you can assign from a cbuffer array

// CHECK-LABEL: define void {{.*}}arr_assign8
// CHECK: [[C:%.*]] = alloca [2 x float], align 4
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[C]], ptr align 4 {{.*}}, i32 8, i1 false)
// CHECK-NEXT: call void @llvm.memcpy.p0.p2.i32(ptr align 4 [[C]], ptr addrspace(2) align 4 @c1, i32 8, i1 false)
// CHECK-NEXT: ret void
void arr_assign8() {
  float C[2] = {1.0, 2.0};
  C = c1;
}

// CHECK-LABEL: define void {{.*}}arr_assign9
// CHECK: [[C:%.*]] = alloca [2 x <4 x i32>], align 16
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 16 [[C]], ptr align 16 {{.*}}, i32 32, i1 false)
// CHECK-NEXT: call void @llvm.memcpy.p0.p2.i32(ptr align 16 [[C]], ptr addrspace(2) align 16 @c2, i32 32, i1 false)
// CHECK-NEXT: ret void
void arr_assign9() {
  int4 C[2] = {1,2,3,4,5,6,7,8};
  C = c2;
}

// CHECK-LABEL: define void {{.*}}arr_assign10
// CHECK: [[C:%.*]] = alloca [2 x [2 x i32]], align 4
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[C]], ptr align 4 {{.*}}, i32 16, i1 false)
// CHECK-NEXT: call void @llvm.memcpy.p0.p2.i32(ptr align 4 [[C]], ptr addrspace(2) align 4 @c3, i32 16, i1 false)
// CHECK-NEXT: ret void
void arr_assign10() {
  int C[2][2] = {1,2,3,4};
  C = c3;
}

// CHECK-LABEL: define void {{.*}}arr_assign11
// CHECK: [[C:%.*]] = alloca [1 x %struct.S], align 1
// CHECK: call void @llvm.memcpy.p0.p2.i32(ptr align 1 [[C]], ptr addrspace(2) align 1 @c4, i32 8, i1 false)
// CHECK-NEXT: ret void
void arr_assign11() {
  S s = {1, 2.0};
  S C[1] = {s};
  C = c4;
}
