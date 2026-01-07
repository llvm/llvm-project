// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -finclude-default-header -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s

struct S {
  int x;
  float f;
};

// CHECK: [[CBLayout:%.*]] = type <{ <{ [1 x <{ float, target("dx.Padding", 12) }>], float }>, target("dx.Padding", 12), [2 x <4 x i32>], <{ [1 x <{ <{ [1 x <{ i32, target("dx.Padding", 12) }>], i32 }>, target("dx.Padding", 12) }>], <{ [1 x <{ i32, target("dx.Padding", 12) }>], i32 }> }>, target("dx.Padding", 12), <{ [1 x <{ %S, target("dx.Padding", 8) }>], %S }> }>

// CHECK: @CBArrays.cb = global target("dx.CBuffer", [[CBLayout]])
// CHECK: @c1 = external hidden addrspace(2) global <{ [1 x <{ float, target("dx.Padding", 12) }>], float }>, align 4
// CHECK: @c2 = external hidden addrspace(2) global [2 x <4 x i32>], align 16
// CHECK: @c3 = external hidden addrspace(2) global <{ [1 x <{ <{ [1 x <{ i32, target("dx.Padding", 12) }>], i32 }>, target("dx.Padding", 12) }>], <{ [1 x <{ i32, target("dx.Padding", 12) }>], i32 }> }>, align 4
// CHECK: @c4 = external hidden addrspace(2) global <{ [1 x <{ %S, target("dx.Padding", 8) }>], %S }>, align 1

cbuffer CBArrays : register(b0) {
  float c1[2];
  int4 c2[2];
  int c3[2][2];
  S c4[2];
}

// CHECK-LABEL: define hidden void {{.*}}arr_assign1
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

// CHECK-LABEL: define hidden void {{.*}}arr_assign2
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

// CHECK-LABEL: define hidden void {{.*}}arr_assign3
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

// CHECK-LABEL: define hidden void {{.*}}arr_assign4
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

// CHECK-LABEL: define hidden void {{.*}}arr_assign5
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

// CHECK-LABEL: define hidden void {{.*}}arr_assign6
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

// CHECK-LABEL: define hidden void {{.*}}arr_assign7
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

// CHECK-LABEL: define hidden void {{.*}}arr_assign8
// CHECK: [[C:%.*]] = alloca [2 x float], align 4
// CHECK-NEXT: [[V0:%.*]] = getelementptr inbounds [2 x float], ptr [[C]], i32 0
// CHECK-NEXT: [[L0:%.*]] = load float, ptr addrspace(2) @c1, align 4
// CHECK-NEXT: store float [[L0]], ptr [[V0]], align 4
// CHECK-NEXT: [[V1:%.*]] = getelementptr inbounds [2 x float], ptr [[C]], i32 0, i32 1
// CHECK-NEXT: [[L1:%.*]] = load float, ptr addrspace(2) getelementptr inbounds (<{ [1 x <{ float, target("dx.Padding", 12) }>], float }>, ptr addrspace(2) @c1, i32 0, i32 1), align 4
// CHECK-NEXT: store float [[L1]], ptr [[V1]], align 4
// CHECK-NEXT: ret void
void arr_assign8() {
  float C[2];
  C = c1;
}

// TODO: We should be able to just memcpy here.
// See https://github.com/llvm/wg-hlsl/issues/351
//
// CHECK-LABEL: define hidden void {{.*}}arr_assign9
// CHECK: [[C:%.*]] = alloca [2 x <4 x i32>], align 16
// CHECK-NEXT: [[V0:%.*]] = getelementptr inbounds [2 x <4 x i32>], ptr [[C]], i32 0
// CHECK-NEXT: [[L0:%.*]] = load <4 x i32>, ptr addrspace(2) @c2, align 16
// CHECK-NEXT: store <4 x i32> [[L0]], ptr [[V0]], align 16
// CHECK-NEXT: [[V1:%.*]] = getelementptr inbounds [2 x <4 x i32>], ptr [[C]], i32 0, i32 1
// CHECK-NEXT: [[L1:%.*]] = load <4 x i32>, ptr addrspace(2) getelementptr inbounds ([2 x <4 x i32>], ptr addrspace(2) @c2, i32 0, i32 1), align 16
// CHECK-NEXT: store <4 x i32> [[L1]], ptr [[V1]], align 16
// CHECK-NEXT: ret void
void arr_assign9() {
  int4 C[2];
  C = c2;
}

// CHECK-LABEL: define hidden void {{.*}}arr_assign10
// CHECK: [[C:%.*]] = alloca [2 x [2 x i32]], align 4
// CHECK-NEXT: [[V0:%.*]] = getelementptr inbounds [2 x [2 x i32]], ptr [[C]], i32 0, i32 0, i32 0
// CHECK-NEXT: [[L0:%.*]] = load i32, ptr addrspace(2) @c3, align 4
// CHECK-NEXT: store i32 [[L0]], ptr [[V0]], align 4
// CHECK-NEXT: [[V1:%.*]] = getelementptr inbounds [2 x [2 x i32]], ptr [[C]], i32 0, i32 0, i32 1
// CHECK-NEXT: [[L1:%.*]] = load i32, ptr addrspace(2) getelementptr inbounds (<{ [1 x <{ <{ [1 x <{ i32, target("dx.Padding", 12) }>], i32 }>, target("dx.Padding", 12) }>], <{ [1 x <{ i32, target("dx.Padding", 12) }>], i32 }> }>, ptr addrspace(2) @c3, i32 0, i32 0, i32 0, i32 0, i32 1), align 4
// CHECK-NEXT: store i32 [[L1]], ptr [[V1]], align 4
// CHECK-NEXT: [[V2:%.*]] = getelementptr inbounds [2 x [2 x i32]], ptr [[C]], i32 0, i32 1, i32 0
// CHECK-NEXT: [[L2:%.*]] = load i32, ptr addrspace(2) getelementptr inbounds (<{ [1 x <{ <{ [1 x <{ i32, target("dx.Padding", 12) }>], i32 }>, target("dx.Padding", 12) }>], <{ [1 x <{ i32, target("dx.Padding", 12) }>], i32 }> }>, ptr addrspace(2) @c3, i32 0, i32 1, i32 0, i32 0, i32 0), align 4
// CHECK-NEXT: store i32 [[L2]], ptr [[V2]], align 4
// CHECK-NEXT: [[V3:%.*]] = getelementptr inbounds [2 x [2 x i32]], ptr [[C]], i32 0, i32 1, i32 1
// CHECK-NEXT: [[L3:%.*]] = load i32, ptr addrspace(2) getelementptr inbounds (<{ [1 x <{ <{ [1 x <{ i32, target("dx.Padding", 12) }>], i32 }>, target("dx.Padding", 12) }>], <{ [1 x <{ i32, target("dx.Padding", 12) }>], i32 }> }>, ptr addrspace(2) @c3, i32 0, i32 1, i32 1), align 4
// CHECK-NEXT: store i32 [[L3]], ptr [[V3]], align 4
// CHECK-NEXT: ret void
void arr_assign10() {
  int C[2][2];
  C = c3;
}

// CHECK-LABEL: define hidden void {{.*}}arr_assign11
// CHECK: [[C:%.*]] = alloca [2 x %struct.S], align 1
// CHECK-NEXT: [[V0:%.*]] = getelementptr inbounds [2 x %struct.S], ptr [[C]], i32 0, i32 0, i32 0
// CHECK-NEXT: [[L0:%.*]] = load i32, ptr addrspace(2) @c4, align 4
// CHECK-NEXT: store i32 [[L0]], ptr [[V0]], align 4
// CHECK-NEXT: [[V1:%.*]] = getelementptr inbounds [2 x %struct.S], ptr [[C]], i32 0, i32 0, i32 1
// CHECK-NEXT: [[L1:%.*]] = load float, ptr addrspace(2) getelementptr inbounds (<{ [1 x <{ %S, target("dx.Padding", 8) }>], %S }>, ptr addrspace(2) @c4, i32 0, i32 0, i32 0, i32 0, i32 1), align 4
// CHECK-NEXT: store float [[L1]], ptr [[V1]], align 4
// CHECK-NEXT: [[V2:%.*]] = getelementptr inbounds [2 x %struct.S], ptr [[C]], i32 0, i32 1, i32 0
// CHECK-NEXT: [[L2:%.*]] = load i32, ptr addrspace(2) getelementptr inbounds (<{ [1 x <{ %S, target("dx.Padding", 8) }>], %S }>, ptr addrspace(2) @c4, i32 0, i32 1, i32 0), align 4
// CHECK-NEXT: store i32 [[L2]], ptr [[V2]], align 4
// CHECK-NEXT: [[V3:%.*]] = getelementptr inbounds [2 x %struct.S], ptr [[C]], i32 0, i32 1, i32 1
// CHECK-NEXT: [[L3:%.*]] = load float, ptr addrspace(2) getelementptr inbounds (<{ [1 x <{ %S, target("dx.Padding", 8) }>], %S }>, ptr addrspace(2) @c4, i32 0, i32 1, i32 1), align 4
// CHECK-NEXT: store float [[L3]], ptr [[V3]], align 4
// CHECK-NEXT: ret void
void arr_assign11() {
  S C[2];
  C = c4;
}
