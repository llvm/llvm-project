// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.6-compute -finclude-default-header \
// RUN:   -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s

// This test verifies handling of local arrays of resources when used as a function argument.

// Resource array elements are initialized on access; there should never a global
// array of resources (unless it is static).
// CHECK-NOT: @_ZL1A = internal global [3 x %"class.hlsl::RWBuffer"] poison, align 4

// CHECK: [[BufA:@.*]] = private unnamed_addr constant [2 x i8] c"A\00", align 1

RWBuffer<float> A[3] : register(u0);
RWStructuredBuffer<float> Out : register(u0);

// NOTE: _ZNK4hlsl8RWBufferIfEixEj is the subscript operator for RWBuffer<float> and
//       _ZNK4hlsl18RWStructuredBufferIfEixEj is the subscript operator for RWStructuredBuffer<float>

// CHECK: define {{.*}} float @_Z3fooA3_N4hlsl8RWBufferIfEE(ptr noundef byval([3 x %"class.hlsl::RWBuffer"]) align 4 %LocalA)
// CHECK-NEXT: entry:
// CHECK-NEXT: %[[#C_ENTRY:]] = call token @llvm.experimental.convergence.entry()
float foo(RWBuffer<float> LocalA[3]) {
// CHECK-NEXT: %[[LocalA_2_Ptr:.*]] = getelementptr inbounds [3 x %"class.hlsl::RWBuffer"], ptr %LocalA, i32 0, i32 2
// CHECK-NEXT: %[[BufPtr:.*]] = call {{.*}} ptr @_ZNK4hlsl8RWBufferIfEixEj(ptr {{.*}} %[[LocalA_2_Ptr]], i32 noundef 0)
// CHECK-NEXT: %[[Value:.*]] = load float, ptr %[[BufPtr]], align 4
// CHECK-NEXT: ret float %[[Value]]
  return LocalA[2][0];
}

// CHECK: define internal void @_Z4mainv()
// CHECK-NEXT: entry:
// CHECK-NEXT: %[[#C_ENTRY:]] = call token @llvm.experimental.convergence.entry()
[numthreads(4,1,1)]
void main() {
// Check that the `main` function calls `foo` with a local copy of the array
// CHECK-NEXT: [[Tmp:%.*]] = alloca [3 x %"class.hlsl::RWBuffer"], align 4
// CHECK-NEXT: [[TmpPtr0:%.*]] = getelementptr [3 x %"class.hlsl::RWBuffer"], ptr [[Tmp]], i32 0, i32 0
// CHECK-NEXT: call void @_ZN4hlsl8RWBufferIfE19__createFromBindingEjjijPKc(ptr {{.*}} [[TmpPtr0]], i32 noundef 0, i32 noundef 0, i32 noundef 3, i32 noundef 0, ptr noundef [[BufA]])
// CHECK-NEXT: [[TmpPtr1:%.*]] = getelementptr [3 x %"class.hlsl::RWBuffer"], ptr [[Tmp]], i32 0, i32 1
// CHECK-NEXT: call void @_ZN4hlsl8RWBufferIfE19__createFromBindingEjjijPKc(ptr {{.*}} [[TmpPtr1]], i32 noundef 0, i32 noundef 0, i32 noundef 3, i32 noundef 1, ptr noundef [[BufA]])
// CHECK-NEXT: [[TmpPtr2:%.*]] = getelementptr [3 x %"class.hlsl::RWBuffer"], ptr [[Tmp]], i32 0, i32 2
// CHECK-NEXT: call void @_ZN4hlsl8RWBufferIfE19__createFromBindingEjjijPKc(ptr {{.*}} [[TmpPtr2]], i32 noundef 0, i32 noundef 0, i32 noundef 3, i32 noundef 2, ptr noundef [[BufA]])

// CHECK-NEXT: %[[ReturnedValue:.*]] = call {{.*}} float @_Z3fooA3_N4hlsl8RWBufferIfEE(ptr {{.*}} [[Tmp]])
// CHECK-NEXT: %[[OutBufPtr:.*]] = call {{.*}} ptr @_ZNK4hlsl18RWStructuredBufferIfEixEj(ptr {{.*}} @_ZL3Out, i32 noundef 0)
// CHECK-NEXT: store float %[[ReturnedValue]], ptr %[[OutBufPtr]], align 4
// CHECK-NEXT: ret void
  Out[0] = foo(A);
}
