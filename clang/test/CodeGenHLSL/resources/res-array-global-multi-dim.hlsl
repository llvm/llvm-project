// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.6-compute -finclude-default-header \
// RUN:   -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s
// RUN: %clang_cc1 -finclude-default-header -triple spirv-unknown-vulkan-compute \
// RUN:   -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s

// CHECK: @[[BufB:.*]] = private unnamed_addr constant [2 x i8] c"B\00", align 1
// CHECK: @[[BufC:.*]] = private unnamed_addr constant [2 x i8] c"C\00", align 1
// CHECK: @[[BufD:.*]] = private unnamed_addr constant [2 x i8] c"D\00", align 1

RWBuffer<float> B[4][4] : register(u2);
RWBuffer<int> C[2][2][5] : register(u10, space1);

typedef RWBuffer<uint> RWBufferArrayTenByFive[10][5]; // test typedef for the resource array type
RWBufferArrayTenByFive D; // implicit binding -> u18, space0

RWStructuredBuffer<float> Out;

[numthreads(4,1,1)]
void main() {
  // CHECK: define internal{{.*}} void @_Z4mainv()
  // CHECK: %[[Tmp0:.*]] = alloca %"class.hlsl::RWBuffer
  // CHECK: %[[Tmp1:.*]] = alloca %"class.hlsl::RWBuffer
  // CHECK: %[[Tmp2:.*]] = alloca %"class.hlsl::RWBuffer
  // CHECK: %[[Tmp3:.*]] = alloca %"class.hlsl::RWBuffer

  // NOTE:
  // Constructor call for explicit binding has "jjij" in the mangled name and the arguments are (register, space, range_size, index, name).
  // For implicit binding the constructor has "jijj" in the mangled name and the arguments are (space, range_size, index, order_id, name).
  // The range_size can be -1 for unbounded arrays, and that is the only signed int in the signature.
  // The order_id argument is a sequential number that is assigned to resources with implicit binding and corresponds to the order in which 
  // the resources were declared. It is needed because implicit bindings are assigned later on in an LLVM pass that needs to know the order
  // of the resource declarations.

  // Make sure that B[3][2] is translated to a RWBuffer<float> constructor call for explicit binding (u2, space0) with range 16 and index 14
  // CHECK: call void @_ZN4hlsl8RWBufferIfEC1EjjijPKc(ptr {{.*}} %[[Tmp0]], i32 noundef 2, i32 noundef 0, i32 noundef 16, i32 noundef 14, ptr noundef @[[BufB]])

  // Make sure that C[1][0][3] is translated to a RWBuffer<int> constructor call for explicit binding (u10, space1) with range 20 and index 13
  // CHECK: call void @_ZN4hlsl8RWBufferIiEC1EjjijPKc(ptr {{.*}} %[[Tmp1]], i32 noundef 10, i32 noundef 1, i32 noundef 20, i32 noundef 13, ptr noundef @[[BufC]])

  // Make sure that D[9][2] is translated to a RWBuffer<uint> constructor call for implicit binding (u18, space0) with range 50 and index 47
  // CHECK: call void @_ZN4hlsl8RWBufferIjEC1EjijjPKc(ptr {{.*}} %[[Tmp2]], i32 noundef 0, i32 noundef 50, i32 noundef 47, i32 noundef 0, ptr noundef @[[BufD]])

  // Make sure that the second B[3][2] is translated to the same a RWBuffer<float> constructor call as the first B[3][2] subscript
  // CHECK: call void @_ZN4hlsl8RWBufferIfEC1EjjijPKc(ptr {{.*}} %[[Tmp3]], i32 noundef 2, i32 noundef 0, i32 noundef 16, i32 noundef 14, ptr noundef @[[BufB]])
  Out[0] =  B[3][2][0] + (float)C[1][0][3][0] + (float)D[9][2][0] + B[3][2][1];
}
