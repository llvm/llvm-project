// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.6-compute -finclude-default-header \
// RUN:   -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s

// CHECK: @[[BufA:.*]] = private unnamed_addr constant [2 x i8] c"A\00", align 1

RWBuffer<float> A[4][2] : register(u10, space2);
RWStructuredBuffer<float> Out;

float foo(RWBuffer<float> Arr[2]) {
  return Arr[1][0];
}

// NOTE:
// - _ZN4hlsl8RWBufferIfEC1EjjijPKc is the constructor call for explicit binding
//    (has "jjij" in the mangled name) and the arguments are (register, space, range_size, index, name).
// - _ZN4hlsl8RWBufferIfEixEj is the subscript operator for RWBuffer<float>

// CHECK: define internal void @_Z4mainj(i32 noundef %GI)
// CHECK-NEXT: entry:
// CHECK-NEXT: %[[GI_alloca:.*]] = alloca i32, align 4
// CHECK-NEXT: %Sub = alloca [2 x %"class.hlsl::RWBuffer"], align 4
// CHECK-NEXT: %[[Tmp0:.*]] = alloca [2 x %"class.hlsl::RWBuffer"], align 4
// CHECK-NEXT: %a = alloca float, align 4
// CHECK-NEXT: %b = alloca float, align 4
// CHECK-NEXT: %[[Tmp1:.*]] = alloca [2 x %"class.hlsl::RWBuffer"], align 4
// CHECK-NEXT: %[[Tmp2:.*]] = alloca [2 x %"class.hlsl::RWBuffer"], align 4
// CHECK-NEXT: store i32 %GI, ptr %[[GI_alloca]], align 4
[numthreads(4,1,1)]
void main(uint GI : SV_GroupThreadID) {
// Codegen for "A[2]" - create local array [[Tmp0]] of size 2 and initialize
// each element by a call to the resource constructor
// CHECK-NEXT: %[[Ptr_Tmp0_0:.*]] = getelementptr [2 x %"class.hlsl::RWBuffer"], ptr %[[Tmp0]], i32 0, i32 0
// CHECK-NEXT: call void @_ZN4hlsl8RWBufferIfEC1EjjijPKc(ptr {{.*}} %[[Ptr_Tmp0_0]], i32 noundef 10, i32 noundef 2, i32 noundef 8, i32 noundef 6, ptr noundef @[[BufA]])
// CHECK-NEXT: %[[Ptr_Tmp0_1:.*]] = getelementptr [2 x %"class.hlsl::RWBuffer"], ptr %[[Tmp0]], i32 0, i32 1
// CHECK-NEXT: call void @_ZN4hlsl8RWBufferIfEC1EjjijPKc(ptr {{.*}} %[[Ptr_Tmp0_1]], i32 noundef 10, i32 noundef 2, i32 noundef 8, i32 noundef 7, ptr noundef @[[BufA]])
// After this Tmp0 values are copied to %Sub using the standard array loop initializaion
// (generated from ArrayInitLoopExpr AST node)
  RWBuffer<float> Sub[2] = A[3];

// CHECK: %[[Ptr_Sub_1:.*]] = getelementptr inbounds [2 x %"class.hlsl::RWBuffer"], ptr %Sub, i32 0, i32 1
// CHECK-NEXT: %[[BufPtr:.*]] = call {{.*}} ptr @_ZN4hlsl8RWBufferIfEixEj(ptr {{.*}} %[[Ptr_Sub_1]], i32 noundef 0)
// CHECK-NEXT: %[[Sub_1_0_Value:.*]] = load float, ptr %[[BufPtr]], align 4
// CHECK-NEXT: store float %[[Sub_1_0_Value]], ptr %a, align 4
  float a = Sub[1][0];

// Codegen for "foo(A[GI])" - create local array [[Tmp2]] of size 2 and initialize
// each element by a call to the resource constructor with dynamic index, and then
// copy-in the array as an argument of "foo"
// CHECK: %[[GI:.*]] = load i32, ptr %[[GI_alloca]], align 4
// CHECK-NEXT: %[[Index_A_GI_0:.*]] = mul i32 %[[GI]], 2
// CHECK-NEXT: %[[Ptr_Tmp2_GI_0:.*]] = getelementptr [2 x %"class.hlsl::RWBuffer"], ptr %[[Tmp2]], i32 0, i32 0
// CHECK-NEXT: call void @_ZN4hlsl8RWBufferIfEC1EjjijPKc(ptr {{.*}} %[[Ptr_Tmp2_GI_0]], i32 noundef 10, i32 noundef 2, i32 noundef 8, i32 noundef %[[Index_A_GI_0]], ptr noundef @[[BufA]])
// CHECK-NEXT: %[[Index_A_GI_1:.*]] = add i32 %[[Index_A_GI_0]], 1
// CHECK-NEXT: %[[Ptr_Tmp2_GI_1:.*]] = getelementptr [2 x %"class.hlsl::RWBuffer"], ptr %[[Tmp2]], i32 0, i32 1
// CHECK-NEXT: call void @_ZN4hlsl8RWBufferIfEC1EjjijPKc(ptr {{.*}} %[[Ptr_Tmp2_GI_1]], i32 noundef 10, i32 noundef 2, i32 noundef 8, i32 noundef %[[Index_A_GI_1]], ptr noundef @[[BufA]])
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 4 %[[Tmp1]], ptr align 4 %[[Tmp2]], i32 8, i1 false)
// CHECK-NEXT: %[[FooReturned:.*]] = call {{.*}} float @_Z3fooA2_N4hlsl8RWBufferIfEE(ptr noundef byval([2 x %"class.hlsl::RWBuffer"]) align 4 %[[Tmp1]])
// CHECK-NEXT: store float %[[FooReturned]], ptr %b, align 4
  float b = foo(A[GI]);

  Out[0] = a + b;
}
