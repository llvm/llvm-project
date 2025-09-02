// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.6-compute -finclude-default-header \
// RUN:   -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s

// This test verifies handling of local arrays of resources when used as a function argument.

// CHECK: @_ZL1A = internal global [3 x %"class.hlsl::RWBuffer"] poison, align 4

RWBuffer<float> A[3] : register(u0);
RWStructuredBuffer<float> Out : register(u0);

// NOTE: _ZN4hlsl8RWBufferIfEixEj is the subscript operator for RWBuffer<float> and
//       _ZN4hlsl18RWStructuredBufferIfEixEj is the subscript operator for RWStructuredBuffer<float>

// CHECK: define {{.*}} float @_Z3fooA3_N4hlsl8RWBufferIfEE(ptr noundef byval([3 x %"class.hlsl::RWBuffer"]) align 4 %LocalA)
// CHECK-NEXT: entry:
float foo(RWBuffer<float> LocalA[3]) {
// CHECK-NEXT: %[[LocalA_2_Ptr:.*]] = getelementptr inbounds [3 x %"class.hlsl::RWBuffer"], ptr %LocalA, i32 0, i32 2
// CHECK-NEXT: %[[BufPtr:.*]] = call {{.*}} ptr @_ZN4hlsl8RWBufferIfEixEj(ptr {{.*}} %[[LocalA_2_Ptr]], i32 noundef 0)
// CHECK-NEXT: %[[Value:.*]] = load float, ptr %[[BufPtr]], align 4
// CHECK-NEXT: ret float %[[Value]]
  return LocalA[2][0];
}

// CHECK: define internal void @_Z4mainv()
// CHECK-NEXT: entry:
[numthreads(4,1,1)]
void main() {
// Check that the `main` function calls `foo` with a local copy of the array
// CHECK-NEXT: %[[Tmp:.*]] = alloca [3 x %"class.hlsl::RWBuffer"], align 4
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 4 %[[Tmp]], ptr align 4 @_ZL1A, i32 12, i1 false)

// CHECK-NEXT: %[[ReturnedValue:.*]] = call {{.*}} float @_Z3fooA3_N4hlsl8RWBufferIfEE(ptr noundef byval([3 x %"class.hlsl::RWBuffer"]) align 4 %[[Tmp]])
// CHECK-NEXT: %[[OutBufPtr:.*]] = call {{.*}} ptr @_ZN4hlsl18RWStructuredBufferIfEixEj(ptr {{.*}} @_ZL3Out, i32 noundef 0)
// CHECK-NEXT: store float %[[ReturnedValue]], ptr %[[OutBufPtr]], align 4
// CHECK-NEXT: ret void
  Out[0] = foo(A);
}
