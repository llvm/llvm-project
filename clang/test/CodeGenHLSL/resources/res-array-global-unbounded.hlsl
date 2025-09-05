// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.6-compute -finclude-default-header \
// RUN:   -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s -check-prefixes=CHECK,DXIL
// RUN: %clang_cc1 -finclude-default-header -triple spirv-unknown-vulkan-compute \
// RUN:   -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s -check-prefixes=CHECK,SPV

// CHECK: @[[BufA:.*]] = private unnamed_addr constant [2 x i8] c"A\00", align 1
// CHECK: @[[BufB:.*]] = private unnamed_addr constant [2 x i8] c"B\00", align 1

RWBuffer<float> A[] : register(u10, space1);
RWBuffer<int> B[][5][4];

RWStructuredBuffer<float> Out;

float foo(RWBuffer<int> Arr[4], uint Index) {
  return (float)Arr[Index][0];
}

// NOTE:
// - _ZN4hlsl8RWBufferIfEC1EjjijPKc is the constructor call for explicit binding for RWBuffer<float>
//    (has "jjij" in the mangled name) and the arguments are (register, space, range_size, index, name).
// - _ZN4hlsl8RWBufferIiEC1EjijjPKc is the constructor call for implicit binding for RWBuffer<int>
//    (has "jijj" in the mangled name) and the arguments are (space, range_size, index, order_id, name).
// - _ZN4hlsl8RWBufferIfEixEj is the subscript operator on RWBuffer<float>

[numthreads(4,1,1)]
void main(uint GI : SV_GroupIndex) {
  // CHECK: define internal {{.*}}void @_Z4mainj(i32 noundef %GI)
  // CHECK: %[[GI_alloca:.*]] = alloca i32, align 4
  // CHECK-NEXT: %a = alloca float, align 4
  // CHECK-NEXT: %[[Tmp0:.*]] = alloca %"class.hlsl::RWBuffer
  // CHECK-NEXT: %b = alloca float, align 4
  // CHECK-NEXT: %[[Tmp1:.*]] = alloca [4 x %"class.hlsl::RWBuffer"]
  // CHECK-NEXT: %[[Tmp2:.*]] = alloca [4 x %"class.hlsl::RWBuffer"]
  // CHECK-NEXT: store i32 %GI, ptr %[[GI_alloca]], align 4

  // Make sure A[100] is translated to a RWBuffer<float> constructor call with range -1 and index 100
  // and explicit binding (u10, space1) 
  // CHECK: call void @_ZN4hlsl8RWBufferIfEC1EjjijPKc(ptr {{.*}} %[[Tmp0]], i32 noundef 10, i32 noundef 1, i32 noundef -1, i32 noundef 100, ptr noundef @A.str)
  // CHECK-NEXT: %[[BufPtr:.*]] = call {{.*}} ptr{{.*}} @_ZN4hlsl8RWBufferIfEixEj(ptr {{.*}} %[[Tmp0]], i32 noundef 0)
  // CHECK-NEXT: %[[Value1:.*]] = load float, ptr{{.*}} %[[BufPtr]], align 4
  // CHECK-NEXT: store float %[[Value1]], ptr %a, align 4
  float a = A[100][0];

  // Make sure B[2][3] is translated to a local RWBuffer<int>[4] array where each array element
  // is initialized by a constructor call with range -1 and index 52-55 and implicit binding 
  // (space 0, order_id 0) 
  // The first index is calculated from the array dimensions (unbounded x 5 x 4) and indices (2, 3)
  // as 2 * 5 * 4 + 3 * 4 = 52 and the following indices are sequential.
  // CHECK-NEXT: %[[Ptr_Tmp2_0:.*]] = getelementptr [4 x %"class.hlsl::RWBuffer"], ptr %[[Tmp2]], i32 0, i32 0
  // CHECK-NEXT: call void @_ZN4hlsl8RWBufferIiEC1EjijjPKc(ptr {{.*}} %[[Ptr_Tmp2_0]], i32 noundef 0, i32 noundef -1, i32 noundef 52, i32 noundef 0, ptr noundef @B.str)
  // CHECK-NEXT: %[[Ptr_Tmp2_1:.*]] = getelementptr [4 x %"class.hlsl::RWBuffer"], ptr %[[Tmp2]], i32 0, i32 1
  // CHECK-NEXT: call void @_ZN4hlsl8RWBufferIiEC1EjijjPKc(ptr {{.*}} %[[Ptr_Tmp2_1]], i32 noundef 0, i32 noundef -1, i32 noundef 53, i32 noundef 0, ptr noundef @B.str)
  // CHECK-NEXT: %[[Ptr_Tmp2_2:.*]] = getelementptr [4 x %"class.hlsl::RWBuffer"], ptr %[[Tmp2]], i32 0, i32 2
  // CHECK-NEXT: call void @_ZN4hlsl8RWBufferIiEC1EjijjPKc(ptr {{.*}} %[[Ptr_Tmp2_2]], i32 noundef 0, i32 noundef -1, i32 noundef 54, i32 noundef 0, ptr noundef @B.str)
  // CHECK-NEXT: %[[Ptr_Tmp2_3:.*]] = getelementptr [4 x %"class.hlsl::RWBuffer"], ptr %[[Tmp2]], i32 0, i32 3
  // CHECK-NEXT: call void @_ZN4hlsl8RWBufferIiEC1EjijjPKc(ptr {{.*}} %[[Ptr_Tmp2_3]], i32 noundef 0, i32 noundef -1, i32 noundef 55, i32 noundef 0, ptr noundef @B.str)
  // DXIL-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 4 %[[Tmp1]], ptr align 4 %[[Tmp2]], i32 16, i1 false)
  // SPV-NEXT: call void @llvm.memcpy.p0.p0.i64(ptr align 8 %[[Tmp1]], ptr align 8 %[[Tmp2]], i64 32, i1 false)
  // CHECK-NEXT: %[[GI:.*]] = load i32, ptr %[[GI_alloca]], align 4
  // DXIL-NEXT: %[[Value2:.*]] = call {{.*}} float @_Z3fooA4_N4hlsl8RWBufferIiEEj(ptr noundef byval([4 x %"class.hlsl::RWBuffer"]) align 4 %[[Tmp1]], i32 noundef %[[GI]])
  // SPV-NEXT: %[[Value2:.*]] = call {{.*}} float @_Z3fooA4_N4hlsl8RWBufferIiEEj(ptr noundef byval([4 x %"class.hlsl::RWBuffer"]) align 8 %[[Tmp1]], i32 noundef %[[GI]])
  // CHECK-NEXT: store float %[[Value2]], ptr %b, align 4
  float b = foo(B[2][3], GI);

  Out[0] = a + b;
}
