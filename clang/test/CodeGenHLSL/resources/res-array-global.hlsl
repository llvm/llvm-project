// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.6-compute -finclude-default-header \
// RUN:   -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s
// RUN: %clang_cc1 -finclude-default-header -triple spirv-unknown-vulkan-compute \
// RUN:   -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s

// CHECK: @[[BufA:.*]] = private unnamed_addr constant [2 x i8] c"A\00", align 1
// CHECK: @[[BufB:.*]] = private unnamed_addr constant [2 x i8] c"B\00", align 1

RWBuffer<float> A[4] : register(u10, space1);
RWBuffer<int> B[5]; // implicit binding -> u0, space0
RWStructuredBuffer<float> Out;

[numthreads(4,1,1)]
void main() {
  // CHECK: define internal{{.*}} void @_Z4mainv()
  // CHECK: %[[Tmp0:.*]] = alloca %"class.hlsl::RWBuffer
  // CHECK: %[[Tmp1:.*]] = alloca %"class.hlsl::RWBuffer

  // Make sure that A[2] is translated to a RWBuffer<float> constructor call for explicit binding (u10, space1) with range 4 and index 2
  // CHECK: call void @_ZN4hlsl8RWBufferIfEC1EjjijPKc(ptr {{.*}} %[[Tmp0]], i32 noundef 10, i32 noundef 1, i32 noundef 4, i32 noundef 2, ptr noundef @[[BufA]])

  // Make sure that A[3] is translated to a RWBuffer<int> constructor call for implicit binding (u0, space0) with range 5 and index 3
  // CHECK: call void @_ZN4hlsl8RWBufferIiEC1EjijjPKc(ptr {{.*}} %[[Tmp1]], i32 noundef 0, i32 noundef 5, i32 noundef 3, i32 noundef 0, ptr noundef @[[BufB]])

  Out[0] = A[2][0] + (float)B[3][0];
}
