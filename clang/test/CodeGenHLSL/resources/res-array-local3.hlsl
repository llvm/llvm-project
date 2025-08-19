// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.6-compute -finclude-default-header \
// RUN:   -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s

// This test verifies handling of local arrays of resources when used
// as a function argument that is modified inside the function.

// CHECK: @_ZL1X = internal global %"class.hlsl::RWBuffer" poison, align 4
// CHECK: @_ZL1Y = internal global %"class.hlsl::RWBuffer" poison, align 4

RWBuffer<int> X : register(u0);
RWBuffer<int> Y : register(u1);

// CHECK: define {{.*}} @_Z6SomeFnA2_N4hlsl8RWBufferIiEEji(
// CHECK-SAME: ptr noundef byval([2 x %"class.hlsl::RWBuffer"]) align 4 %B, i32 noundef %Idx, i32 noundef %Val0)
// CHECK-NEXT: entry:
// CHECK-NEXT: %[[Idx_addr:.*]] = alloca i32, align 4
// CHECK-NEXT: %[[Val0_addr:.*]] = alloca i32, align 4
// CHECK-NEXT: store i32 %Idx, ptr %[[Idx_addr]], align 4
// CHECK-NEXT: store i32 %Val0, ptr %[[Val0_addr]], align 4
void SomeFn(RWBuffer<int> B[2], uint Idx, int Val0) {

// CHECK-NEXT: %[[B_0_Ptr:.*]] = getelementptr inbounds [2 x %"class.hlsl::RWBuffer"], ptr %B, i32 0, i32 0
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 4 %[[B_0_Ptr]], ptr align 4 @_ZL1Y, i32 4, i1 false)
  B[0] = Y;

// NOTE: _ZN4hlsl8RWBufferIiEixEj is the subscript operator for RWBuffer<int>

// CHECK-NEXT: %[[Val0:.*]] = load i32, ptr %[[Val0_addr]], align 4
// CHECK-NEXT: %[[B_0_Ptr:.*]] = getelementptr inbounds [2 x %"class.hlsl::RWBuffer"], ptr %B, i32 0, i32 0
// CHECK-NEXT: %[[Idx:.*]] = load i32, ptr %[[Idx_addr]], align 4
// CHECK-NEXT: %[[BufPtr:.*]] = call {{.*}} ptr @_ZN4hlsl8RWBufferIiEixEj(ptr {{.*}} %[[B_0_Ptr]], i32 noundef %[[Idx]])
// CHECK-NEXT: store i32 %[[Val0]], ptr %[[BufPtr]], align 4
  B[0][Idx] = Val0;
}

// CHECK: define {{.*}} void @_Z4mainj(i32 noundef %GI)
// CHECK-NEXT: entry:
// CHECK-NEXT: %[[GI_addr:.*]] = alloca i32, align 4
[numthreads(4,1,1)]
void main(uint GI : SV_GroupIndex) {
// CHECK-NEXT: %A = alloca [2 x %"class.hlsl::RWBuffer"], align 4
// CHECK-NEXT: %[[Tmp:.*]] = alloca [2 x %"class.hlsl::RWBuffer"], align 4
// CHECK-NEXT: store i32 %GI, ptr %GI.addr, align 4

// Initialization of array A with resources X and Y
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 4 %A, ptr align 4 @_ZL1X, i32 4, i1 false)
// CHECK-NEXT: %[[A_1_Ptr:.*]] = getelementptr inbounds %"class.hlsl::RWBuffer", ptr %A, i32 1
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 4 %[[A_1_Ptr]], ptr align 4 @_ZL1Y, i32 4, i1 false)
  RWBuffer<int> A[2] = {X, Y};

// Verify that SomeFn is called with a local copy of the array A
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 4 %[[Tmp]], ptr align 4 %A, i32 8, i1 false)
// CHECK-NEXT: %[[GI:.*]] = load i32, ptr %[[GI_addr]], align 4
// CHECK-NEXT: call void @_Z6SomeFnA2_N4hlsl8RWBufferIiEEji(ptr noundef byval([2 x %"class.hlsl::RWBuffer"]) align 4 %[[Tmp]], i32 noundef %[[GI]], i32 noundef 1)
  SomeFn(A, GI, 1);

// CHECK-NEXT: %[[A_0_Ptr:.*]] = getelementptr inbounds [2 x %"class.hlsl::RWBuffer"], ptr %A, i32 0, i32 0
// CHECK-NEXT: %[[GI:.*]] = load i32, ptr %[[GI_addr]], align 4
// CHECK-NEXT: %[[BufPtr:.*]] = call {{.*}} ptr @_ZN4hlsl8RWBufferIiEixEj(ptr {{.*}} %[[A_0_Ptr]], i32 noundef %[[GI]])
// CHECK-NEXT: store i32 2, ptr %[[BufPtr]], align 4
  A[0][GI] = 2;
}
