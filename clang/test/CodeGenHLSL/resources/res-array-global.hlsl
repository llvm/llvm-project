// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.6-compute -finclude-default-header \
// RUN:   -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s -check-prefixes=CHECK,DXIL
// RUN: %clang_cc1 -finclude-default-header -triple spirv-unknown-vulkan-compute \
// RUN:   -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s -check-prefixes=CHECK,SPV

// CHECK: @[[BufA:.*]] = private unnamed_addr constant [2 x i8] c"A\00", align 1
// CHECK: @[[BufB:.*]] = private unnamed_addr constant [2 x i8] c"B\00", align 1
// CHECK: @[[BufC:.*]] = private unnamed_addr constant [2 x i8] c"C\00", align 1
// CHECK: @[[BufD:.*]] = private unnamed_addr constant [2 x i8] c"D\00", align 1
// CHECK: @[[BufE:.*]] = private unnamed_addr constant [2 x i8] c"E\00", align 1

// different explicit binding for DXIL and SPIR-V
[[vk::binding(12, 2)]]
RWBuffer<float> A[4] : register(u10, space1);

[[vk::binding(13)]] // SPIR-V explicit binding 13, set 0
RWBuffer<int> B[5]; // DXIL implicit binding in space0

// same explicit binding for both DXIL and SPIR-V
// (SPIR-V takes the binding from register annotation if there is no vk::binding attribute))
RWBuffer<int> C[3] : register(u2);

// implicit binding for both DXIL and SPIR-V in space/set 0 
RWBuffer<double> D[10];

// implicit binding for both DXIL and SPIR-V with specified space/set 0 
RWBuffer<uint> E[15] : register(space2);

RWStructuredBuffer<float> Out;

[numthreads(4,1,1)]
void main() {
  // CHECK: define internal{{.*}} void @_Z4mainv()
  // CHECK: %[[Tmp0:.*]] = alloca %"class.hlsl::RWBuffer
  // CHECK: %[[Tmp1:.*]] = alloca %"class.hlsl::RWBuffer
  // CHECK: %[[Tmp2:.*]] = alloca %"class.hlsl::RWBuffer
  // CHECK: %[[Tmp3:.*]] = alloca %"class.hlsl::RWBuffer
  // CHECK: %[[Tmp4:.*]] = alloca %"class.hlsl::RWBuffer

  // NOTE:
  // Constructor call for explicit binding has "jjij" in the mangled name and the arguments are (register, space, range_size, index, name).
  // For implicit binding the constructor has "jijj" in the mangled name and the arguments are (space, range_size, index, order_id, name).
  // The range_size can be -1 for unbounded arrays, and that is the only signed int in the signature.
  // The order_id argument is a sequential number that is assigned to resources with implicit binding and corresponds to the order in which 
  // the resources were declared. It is needed because implicit bindings are assigned later on in an LLVM pass that needs to know the order
  // of the resource declarations.

  // Make sure A[2] is translated to a RWBuffer<float> constructor call with range 4 and index 2
  // and DXIL explicit binding (u10, space1)
  // and SPIR-V explicit binding (binding 12, set 2) 
  // DXIL: call void @_ZN4hlsl8RWBufferIfEC1EjjijPKc(ptr {{.*}} %[[Tmp0]], i32 noundef 10, i32 noundef 1, i32 noundef 4, i32 noundef 2, ptr noundef @[[BufA]])
  // SPV: call void @_ZN4hlsl8RWBufferIfEC1EjjijPKc(ptr {{.*}} %[[Tmp0]], i32 noundef 12, i32 noundef 2, i32 noundef 4, i32 noundef 2, ptr noundef @[[BufA]])

  // Make sure B[3] is translated to a RWBuffer<int> constructor call with range 5 and index 3
  // and DXIL for implicit binding in space0, order id 0
  // and SPIR-V explicit binding (binding 13, set 0)
  // DXIL: call void @_ZN4hlsl8RWBufferIiEC1EjijjPKc(ptr {{.*}} %[[Tmp1]], i32 noundef 0, i32 noundef 5, i32 noundef 3, i32 noundef 0, ptr noundef @[[BufB]])
  // SPV: call void @_ZN4hlsl8RWBufferIiEC1EjjijPKc(ptr {{.*}} %[[Tmp1]], i32 noundef 13, i32 noundef 0, i32 noundef 5, i32 noundef 3, ptr noundef @[[BufB]])

  // Make sure C[1] is translated to a RWBuffer<int> constructor call with range 3 and index 1
  // and DXIL explicit binding (u2, space0) 
  // and SPIR-V explicit binding (binding 2, set 0)
  // DXIL: call void @_ZN4hlsl8RWBufferIiEC1EjjijPKc(ptr {{.*}} %[[Tmp2]], i32 noundef 2, i32 noundef 0, i32 noundef 3, i32 noundef 1, ptr noundef @[[BufC]])
  // SPV: call void @_ZN4hlsl8RWBufferIiEC1EjjijPKc(ptr {{.*}} %[[Tmp2]], i32 noundef 2, i32 noundef 0, i32 noundef 3, i32 noundef 1, ptr noundef @[[BufC]])

  // Make sure D[7] is translated to a RWBuffer<double> constructor call with implicit binding
  // for both DXIL and SPIR-V
  // DXIL: call void @_ZN4hlsl8RWBufferIdEC1EjijjPKc(ptr {{.*}} %[[Tmp3]], i32 noundef 0, i32 noundef 10, i32 noundef 7, i32 noundef 1, ptr noundef @[[BufD]])
  // SPV: call void @_ZN4hlsl8RWBufferIdEC1EjijjPKc(ptr {{.*}} %[[Tmp3]], i32 noundef 0, i32 noundef 10, i32 noundef 7, i32 noundef 0, ptr noundef @[[BufD]])

  // Make sure E[5][0] is translated to RWBuffer<uint> constructor call with implicit binding and specified space/set 2
  // DXIL: call void @_ZN4hlsl8RWBufferIjEC1EjijjPKc(ptr {{.*}} %[[Tmp4]], i32 noundef 2, i32 noundef 15, i32 noundef 5, i32 noundef 2, ptr noundef @[[BufE]])
  // SPV: call void @_ZN4hlsl8RWBufferIjEC1EjijjPKc(ptr {{.*}} %[[Tmp4]], i32 noundef 2, i32 noundef 15, i32 noundef 5, i32 noundef 1, ptr noundef @[[BufE]])
  Out[0] = A[2][0] + (float)B[3][0] + (float)C[1][0] + (float)D[7][0] + (float)E[5][0];
}
