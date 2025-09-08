// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.6-compute -finclude-default-header \
// RUN:   -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s

// CHECK: @[[BufA:.*]] = private unnamed_addr constant [2 x i8] c"A\00", align 1

RWBuffer<float> A[5][4][3][2] : register(u10, space2);
RWStructuredBuffer<float> Out;

// CHECK: define {{.*}} float @_Z3fooA3_A2_N4hlsl8RWBufferIfEE(ptr noundef byval([3 x [2 x %"class.hlsl::RWBuffer"]]) align 4 %Arr)
// CHECK-NEXT: entry:
float foo(RWBuffer<float> Arr[3][2]) {
// CHECK-NEXT: %[[Arr_1_Ptr:.*]] = getelementptr inbounds [3 x [2 x %"class.hlsl::RWBuffer"]], ptr %Arr, i32 0, i32 1
// CHECK-NEXT: %[[Arr_1_0_Ptr:.*]] = getelementptr inbounds [2 x %"class.hlsl::RWBuffer"], ptr %[[Arr_1_Ptr]], i32 0, i32 0
// CHECK-NEXT: %[[BufPtr:.*]] = call {{.*}} ptr @_ZN4hlsl8RWBufferIfEixEj(ptr {{.*}} %[[Arr_1_0_Ptr]], i32 noundef 0)
// CHECK-NEXT: %[[Value:.*]] = load float, ptr %[[BufPtr]], align 4
// CHECK-NEXT: ret float %[[Value]]
  return Arr[1][0][0];
}

// NOTE:
// - _ZN4hlsl8RWBufferIfEC1EjjijPKc is the constructor call for explicit binding
//    (has "jjij" in the mangled name) and the arguments are (register, space, range_size, index, name).
// - _ZN4hlsl8RWBufferIfEixEj is the subscript operator for RWBuffer<float>

// CHECK: define internal void @_Z4mainj(i32 noundef %GI)
// CHECK-NEXT: entry:
// CHECK-NEXT: %[[GI_alloca:.*]] = alloca i32, align 4
// CHECK-NEXT: %Sub = alloca [3 x [2 x %"class.hlsl::RWBuffer"]], align 4
// CHECK-NEXT: %[[Tmp0:.*]] = alloca [3 x [2 x %"class.hlsl::RWBuffer"]], align 4
// CHECK-NEXT: %a = alloca float, align 4
// CHECK-NEXT: %b = alloca float, align 4
// CHECK-NEXT: %[[Tmp1:.*]] = alloca [3 x [2 x %"class.hlsl::RWBuffer"]], align 4
// CHECK-NEXT: %[[Tmp2:.*]] = alloca [3 x [2 x %"class.hlsl::RWBuffer"]], align 4
// CHECK-NEXT: store i32 %GI, ptr %[[GI_alloca]], align 4
[numthreads(4,1,1)]
void main(uint GI : SV_GroupThreadID) {
// Codegen for "A[4][1]" - create local array [[Tmp0]] of size 3 x 2 and initialize
// each element by a call to the resource constructor
// The resource index for A[4][1][0][0] is 102 = 4 * (4 * 3 * 2) + 1 * (3 * 2)
// (index in the resource array as if it was flattened)
// CHECK-NEXT: %[[Ptr_Tmp0_0_0:.*]] = getelementptr [3 x [2 x %"class.hlsl::RWBuffer"]], ptr %tmp, i32 0, i32 0, i32 0
// CHECK-NEXT: call void @_ZN4hlsl8RWBufferIfEC1EjjijPKc(ptr {{.*}} %[[Ptr_Tmp0_0_0]], i32 noundef 10, i32 noundef 2, i32 noundef 120, i32 noundef 102, ptr noundef @[[BufA]])
// CHECK-NEXT: %[[Ptr_Tmp0_0_1:.*]] = getelementptr [3 x [2 x %"class.hlsl::RWBuffer"]], ptr %tmp, i32 0, i32 0, i32 1
// CHECK-NEXT: call void @_ZN4hlsl8RWBufferIfEC1EjjijPKc(ptr {{.*}} %[[Ptr_Tmp0_0_1]], i32 noundef 10, i32 noundef 2, i32 noundef 120, i32 noundef 103, ptr noundef @[[BufA]])
// CHECK-NEXT: %[[Ptr_Tmp0_1_0:.*]] = getelementptr [3 x [2 x %"class.hlsl::RWBuffer"]], ptr %tmp, i32 0, i32 1, i32 0
// CHECK-NEXT: call void @_ZN4hlsl8RWBufferIfEC1EjjijPKc(ptr {{.*}} %[[Ptr_Tmp0_1_0]], i32 noundef 10, i32 noundef 2, i32 noundef 120, i32 noundef 104, ptr noundef @[[BufA]])
// CHECK-NEXT: %[[Ptr_Tmp0_1_1:.*]] = getelementptr [3 x [2 x %"class.hlsl::RWBuffer"]], ptr %tmp, i32 0, i32 1, i32 1
// CHECK-NEXT: call void @_ZN4hlsl8RWBufferIfEC1EjjijPKc(ptr {{.*}} %[[Ptr_Tmp0_1_1]], i32 noundef 10, i32 noundef 2, i32 noundef 120, i32 noundef 105, ptr noundef @[[BufA]])
// CHECK-NEXT: %[[Ptr_Tmp0_2_0:.*]] = getelementptr [3 x [2 x %"class.hlsl::RWBuffer"]], ptr %tmp, i32 0, i32 2, i32 0
// CHECK-NEXT: call void @_ZN4hlsl8RWBufferIfEC1EjjijPKc(ptr {{.*}} %[[Ptr_Tmp0_2_0]], i32 noundef 10, i32 noundef 2, i32 noundef 120, i32 noundef 106, ptr noundef @[[BufA]])
// CHECK-NEXT: %[[Ptr_Tmp0_2_1:.*]] = getelementptr [3 x [2 x %"class.hlsl::RWBuffer"]], ptr %tmp, i32 0, i32 2, i32 1
// CHECK-NEXT: call void @_ZN4hlsl8RWBufferIfEC1EjjijPKc(ptr {{.*}} %[[Ptr_Tmp0_2_1]], i32 noundef 10, i32 noundef 2, i32 noundef 120, i32 noundef 107, ptr noundef @[[BufA]])
// After this Tmp0 values are copied to %Sub using the standard array loop initializaion
// (generated from ArrayInitLoopExpr AST node)
  RWBuffer<float> Sub[3][2] = A[4][1];

// CHECK: %[[Ptr_Sub_2:.*]] = getelementptr inbounds  [3 x [2 x %"class.hlsl::RWBuffer"]], ptr %Sub, i32 0, i32 2
// CHECK: %[[Ptr_Sub_2_1:.*]] = getelementptr inbounds [2 x %"class.hlsl::RWBuffer"], ptr %[[Ptr_Sub_2]], i32 0, i32 1
// CHECK-NEXT: %[[BufPtr:.*]] = call {{.*}} ptr @_ZN4hlsl8RWBufferIfEixEj(ptr {{.*}} %[[Ptr_Sub_2_1]], i32 noundef 0)
// CHECK-NEXT: %[[Sub_2_1_0_Value:.*]] = load float, ptr %[[BufPtr]], align 4
// CHECK-NEXT: store float %[[Sub_2_1_0_Value]], ptr %a, align 4
  float a = Sub[2][1][0];

// Codegen for "foo(A[2][GI])" - create local array [[Tmp2]] of size 3 x 2 and initialize
// each element by a call to the resource constructor with dynamic index, and then
// copy-in the array as an argument of "foo"

// Calculate the resource index for A[2][GI][0][0] (index in the resource array as if it was flattened)
// The index is 2 * (4 * 3 * 2) + GI * (3 * 2) = 48 + GI * 6
// CHECK: %[[GI:.*]] = load i32, ptr %[[GI_alloca]], align 4
// CHECK-NEXT: %[[Index_A_2_GI_Tmp:.*]] = mul i32 %[[GI]], 6
// CHECK-NEXT: %[[Index_A_2_GI_0_0:.*]] = add i32 %[[Index_A_2_GI_Tmp]], 48

// A[2][GI][0][0]
// CHECK-NEXT: %[[Ptr_Tmp2_0_0:.*]] = getelementptr [3 x [2 x %"class.hlsl::RWBuffer"]], ptr %[[Tmp2]], i32 0, i32 0, i32 0
// CHECK-NEXT: call void @_ZN4hlsl8RWBufferIfEC1EjjijPKc(ptr {{.*}} %[[Ptr_Tmp2_0_0]], i32 noundef 10, i32 noundef 2, i32 noundef 120, i32 noundef %[[Index_A_2_GI_0_0]], ptr noundef @[[BufA]])

// A[2][GI][0][1]
// CHECK-NEXT: %[[Index_A_2_GI_0_1:.*]] = add i32 %[[Index_A_2_GI_0_0]], 1
// CHECK-NEXT: %[[Ptr_Tmp2_0_1:.*]] = getelementptr [3 x [2 x %"class.hlsl::RWBuffer"]], ptr %[[Tmp2]], i32 0, i32 0, i32 1
// CHECK-NEXT: call void @_ZN4hlsl8RWBufferIfEC1EjjijPKc(ptr {{.*}} %[[Ptr_Tmp2_0_1]], i32 noundef 10, i32 noundef 2, i32 noundef 120, i32 noundef %[[Index_A_2_GI_0_1]], ptr noundef @[[BufA]])

// A[2][GI][1][0]
// CHECK-NEXT: %[[Index_A_2_GI_1_0:.*]] = add i32 %[[Index_A_2_GI_0_1]], 1
// CHECK-NEXT: %[[Ptr_Tmp2_1_0:.*]] = getelementptr [3 x [2 x %"class.hlsl::RWBuffer"]], ptr %[[Tmp2]], i32 0, i32 1, i32 0
// CHECK-NEXT: call void @_ZN4hlsl8RWBufferIfEC1EjjijPKc(ptr {{.*}} %[[Ptr_Tmp2_1_0]], i32 noundef 10, i32 noundef 2, i32 noundef 120, i32 noundef %[[Index_A_2_GI_1_0]], ptr noundef @[[BufA]])

// A[2][GI][1][1]
// CHECK-NEXT: %[[Index_A_2_GI_1_1:.*]] = add i32 %[[Index_A_2_GI_1_0]], 1
// CHECK-NEXT: %[[Ptr_Tmp2_1_1:.*]] = getelementptr [3 x [2 x %"class.hlsl::RWBuffer"]], ptr %[[Tmp2]], i32 0, i32 1, i32 1
// CHECK-NEXT: call void @_ZN4hlsl8RWBufferIfEC1EjjijPKc(ptr {{.*}} %[[Ptr_Tmp2_1_1]], i32 noundef 10, i32 noundef 2, i32 noundef 120, i32 noundef %[[Index_A_2_GI_1_1]], ptr noundef @[[BufA]])

// A[2][GI][2][0]
// CHECK-NEXT: %[[Index_A_2_GI_2_0:.*]] = add i32 %[[Index_A_2_GI_1_1]], 1
// CHECK-NEXT: %[[Ptr_Tmp2_2_0:.*]] = getelementptr [3 x [2 x %"class.hlsl::RWBuffer"]], ptr %[[Tmp2]], i32 0, i32 2, i32 0
// CHECK-NEXT: call void @_ZN4hlsl8RWBufferIfEC1EjjijPKc(ptr {{.*}} %[[Ptr_Tmp2_2_0]], i32 noundef 10, i32 noundef 2, i32 noundef 120, i32 noundef %[[Index_A_2_GI_2_0]], ptr noundef @[[BufA]])

// A[2][GI][2][1]
// CHECK-NEXT: %[[Index_A_2_GI_2_1:.*]] = add i32 %[[Index_A_2_GI_2_0]], 1
// CHECK-NEXT: %[[Ptr_Tmp2_2_1:.*]] = getelementptr [3 x [2 x %"class.hlsl::RWBuffer"]], ptr %[[Tmp2]], i32 0, i32 2, i32 1
// CHECK-NEXT: call void @_ZN4hlsl8RWBufferIfEC1EjjijPKc(ptr {{.*}} %[[Ptr_Tmp2_2_1]], i32 noundef 10, i32 noundef 2, i32 noundef 120, i32 noundef %[[Index_A_2_GI_2_1]], ptr noundef @[[BufA]])

// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 4 %[[Tmp1]], ptr align 4 %[[Tmp2]], i32 24, i1 false)
// CHECK-NEXT: %[[FooReturned:.*]] = call {{.*}} float @_Z3fooA3_A2_N4hlsl8RWBufferIfEE(ptr noundef byval([3 x [2 x %"class.hlsl::RWBuffer"]]) align 4 %[[Tmp1]])
// CHECK-NEXT: store float %[[FooReturned]], ptr %b, align 4
  float b = foo(A[2][GI]);

  Out[0] = a + b;
}
