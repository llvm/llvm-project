// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.6-compute -finclude-default-header \
// RUN:   -emit-llvm -disable-llvm-passes -o - %s | llvm-cxxfilt | FileCheck %s

// CHECK: @[[BufA:.*]] = private unnamed_addr constant [2 x i8] c"A\00", align 1

RWBuffer<float> A[4][2] : register(u10, space2);
RWStructuredBuffer<float> Out;

float foo(RWBuffer<float> Arr[2]) {
  return Arr[1][0];
}

// CHECK: define internal void @main(unsigned int)(i32 noundef %GI)
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
// each element by a call to RWBuffer<float>::__createFromBinding method
// CHECK-NEXT: %[[Ptr_Tmp0_0:.*]] = getelementptr [2 x %"class.hlsl::RWBuffer"], ptr %[[Tmp0]], i32 0, i32 0
// CHECK-NEXT: call void @hlsl::RWBuffer<float>::__createFromBinding(unsigned int, unsigned int, int, unsigned int, char const*)
// CHECK-SAME: (ptr {{.*}} sret(%"class.hlsl::RWBuffer") align 4 %[[Ptr_Tmp0_0]],
// CHECK-SAME: i32 noundef 10, i32 noundef 2, i32 noundef 8, i32 noundef 6, ptr noundef @[[BufA]])
// CHECK-NEXT: %[[Ptr_Tmp0_1:.*]] = getelementptr [2 x %"class.hlsl::RWBuffer"], ptr %[[Tmp0]], i32 0, i32 1
// CHECK-NEXT: call void @hlsl::RWBuffer<float>::__createFromBinding(unsigned int, unsigned int, int, unsigned int, char const*)
// CHECK-SAME: (ptr {{.*}} sret(%"class.hlsl::RWBuffer") align 4 %[[Ptr_Tmp0_1]],
// CHECK-SAME: i32 noundef 10, i32 noundef 2, i32 noundef 8, i32 noundef 7, ptr noundef @[[BufA]])
// After this Tmp0 values are copied to %Sub using the standard array loop initializaion
// (generated from ArrayInitLoopExpr AST node)
  RWBuffer<float> Sub[2] = A[3];

// CHECK: %[[Ptr_Sub_1:.*]] = getelementptr inbounds [2 x %"class.hlsl::RWBuffer"], ptr %Sub, i32 0, i32 1
// CHECK-NEXT: %[[BufPtr:.*]] = call {{.*}} ptr @hlsl::RWBuffer<float>::operator[](unsigned int)(ptr {{.*}} %[[Ptr_Sub_1]], i32 noundef 0)
// CHECK-NEXT: %[[Sub_1_0_Value:.*]] = load float, ptr %[[BufPtr]], align 4
// CHECK-NEXT: store float %[[Sub_1_0_Value]], ptr %a, align 4
  float a = Sub[1][0];

// Codegen for "foo(A[GI])" - create local array [[Tmp2]] of size 2 and initialize
// each element by a call to the RWBuffer<float>::__createFromBinding method 
// with dynamic index, and then copy-in the array as an argument of "foo"
// CHECK: %[[GI:.*]] = load i32, ptr %[[GI_alloca]], align 4
// CHECK-NEXT: %[[Index_A_GI_0:.*]] = mul i32 %[[GI]], 2
// CHECK-NEXT: %[[Ptr_Tmp2_GI_0:.*]] = getelementptr [2 x %"class.hlsl::RWBuffer"], ptr %[[Tmp2]], i32 0, i32 0
// CHECK-NEXT: call void @hlsl::RWBuffer<float>::__createFromBinding(unsigned int, unsigned int, int, unsigned int, char const*)
// CHECK-SAME: (ptr {{.*}} sret(%"class.hlsl::RWBuffer") align 4 %[[Ptr_Tmp2_GI_0]],
// CHECK-SAME: i32 noundef 10, i32 noundef 2, i32 noundef 8, i32 noundef %[[Index_A_GI_0]], ptr noundef @[[BufA]])
// CHECK-NEXT: %[[Index_A_GI_1:.*]] = add i32 %[[Index_A_GI_0]], 1
// CHECK-NEXT: %[[Ptr_Tmp2_GI_1:.*]] = getelementptr [2 x %"class.hlsl::RWBuffer"], ptr %[[Tmp2]], i32 0, i32 1
// CHECK-NEXT: call void @hlsl::RWBuffer<float>::__createFromBinding(unsigned int, unsigned int, int, unsigned int, char const*)
// CHECK-SAME: (ptr {{.*}} sret(%"class.hlsl::RWBuffer") align 4 %[[Ptr_Tmp2_GI_1]],
// CHECK-SAME: i32 noundef 10, i32 noundef 2, i32 noundef 8, i32 noundef %[[Index_A_GI_1]], ptr noundef @[[BufA]])
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 4 %[[Tmp1]], ptr align 4 %[[Tmp2]], i32 8, i1 false)
// CHECK-NEXT: %[[FooReturned:.*]] = call {{.*}} float @foo(hlsl::RWBuffer<float> [2])(ptr noundef byval([2 x %"class.hlsl::RWBuffer"]) align 4 %[[Tmp1]])
// CHECK-NEXT: store float %[[FooReturned]], ptr %b, align 4
  float b = foo(A[GI]);

  Out[0] = a + b;
}
