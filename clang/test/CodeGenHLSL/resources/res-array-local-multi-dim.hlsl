// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.6-compute -finclude-default-header \
// RUN:   -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s

// This test verifies handling of multi-dimensional local arrays of resources
// when used as a function argument and local variable.

// CHECK: @_ZL1A = internal global %"class.hlsl::RWBuffer" poison, align 4
// CHECK: @_ZL1B = internal global %"class.hlsl::RWBuffer" poison, align 4

RWBuffer<float> A : register(u10);
RWBuffer<float> B : register(u20);
RWStructuredBuffer<float> Out;

// NOTE: _ZN4hlsl8RWBufferIfEixEj is the subscript operator for RWBuffer<float> and
//       _ZN4hlsl18RWStructuredBufferIfEixEj is the subscript operator for RWStructuredBuffer<float>

// CHECK: define {{.*}} float @_Z3fooA2_A2_N4hlsl8RWBufferIfEE(ptr noundef byval([2 x [2 x %"class.hlsl::RWBuffer"]]) align 4 %Arr)
// CHECK-NEXT: entry:
float foo(RWBuffer<float> Arr[2][2]) {
// CHECK-NEXT: %[[Arr_1_Ptr:.*]] = getelementptr inbounds [2 x [2 x %"class.hlsl::RWBuffer"]], ptr %Arr, i32 0, i32 1
// CHECK-NEXT: %[[Arr_1_1_Ptr:.*]] = getelementptr inbounds [2 x %"class.hlsl::RWBuffer"], ptr %[[Arr_1_Ptr]], i32 0, i32 1
// CHECK-NEXT: %[[BufPtr:.*]] = call {{.*}} ptr @_ZN4hlsl8RWBufferIfEixEj(ptr {{.*}} %[[Arr_1_1_Ptr]], i32 noundef 0)
// CHECK-NEXT: %[[Value:.*]] = load float, ptr %[[BufPtr]], align 4
// CHECK-NEXT: ret float %[[Value]]
  return Arr[1][1][0];
}

// CHECK: define internal void @_Z4mainv()
// CHECK-NEXT: entry:
[numthreads(4,1,1)]
void main() {
// CHECK-NEXT: %L = alloca [2 x [2 x %"class.hlsl::RWBuffer"]], align 4
// CHECK-NEXT: %[[Tmp:.*]] = alloca [2 x [2 x %"class.hlsl::RWBuffer"]], align 4
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 4 %L, ptr align 4 @_ZL1A, i32 4, i1 false)
// CHECK-NEXT: %[[Ptr1:.*]] = getelementptr inbounds %"class.hlsl::RWBuffer", ptr %L, i32 1
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 4 %[[Ptr1]], ptr align 4 @_ZL1B, i32 4, i1 false)
// CHECK-NEXT: %[[Ptr2:.*]] = getelementptr inbounds [2 x %"class.hlsl::RWBuffer"], ptr %L, i32 1
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 4 %[[Ptr2]], ptr align 4 @_ZL1A, i32 4, i1 false)
// CHECK-NEXT: %[[Ptr3:.*]] = getelementptr inbounds %"class.hlsl::RWBuffer", ptr %[[Ptr2]], i32 1
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 4 %[[Ptr3]], ptr align 4 @_ZL1B, i32 4, i1 false)
  RWBuffer<float> L[2][2] = { { A, B }, { A, B } };

// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 4 %[[Tmp]], ptr align 4 %L, i32 16, i1 false)
// CHECK-NEXT: %[[ReturnedValue:.*]] = call {{.*}}float @_Z3fooA2_A2_N4hlsl8RWBufferIfEE(ptr noundef byval([2 x [2 x %"class.hlsl::RWBuffer"]]) align 4 %[[Tmp]])
// CHECK-NEXT: %[[OutBufPtr:.*]] = call {{.*}} ptr @_ZN4hlsl18RWStructuredBufferIfEixEj(ptr {{.*}} @_ZL3Out, i32 noundef 0)
// CHECK-NEXT: store float %[[ReturnedValue]], ptr %[[OutBufPtr]], align 4
// CHECK-NEXT: ret void
  Out[0] = foo(L);
}
