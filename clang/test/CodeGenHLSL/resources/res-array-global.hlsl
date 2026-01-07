// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.6-compute -finclude-default-header \
// RUN:   -emit-llvm -disable-llvm-passes -o - %s | llvm-cxxfilt | FileCheck %s -check-prefixes=CHECK,DXIL
// RUN: %clang_cc1 -finclude-default-header -triple spirv-unknown-vulkan-compute \
// RUN:   -emit-llvm -disable-llvm-passes -o - %s | llvm-cxxfilt | FileCheck %s -check-prefixes=CHECK,SPV

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
  // CHECK: define internal{{.*}} void @main()
  // CHECK: %[[Tmp0:.*]] = alloca %"class.hlsl::RWBuffer
  // CHECK: %[[Tmp1:.*]] = alloca %"class.hlsl::RWBuffer
  // CHECK: %[[Tmp2:.*]] = alloca %"class.hlsl::RWBuffer
  // CHECK: %[[Tmp3:.*]] = alloca %"class.hlsl::RWBuffer
  // CHECK: %[[Tmp4:.*]] = alloca %"class.hlsl::RWBuffer

  // Make sure A[2] is translated to a RWBuffer<float>::__createFromBinding call with range 4 and index 2
  // and DXIL explicit binding (u10, space1)
  // and SPIR-V explicit binding (binding 12, set 2) 
  // DXIL: call void @hlsl::RWBuffer<float>::__createFromBinding(unsigned int, unsigned int, int, unsigned int, char const*)
  // DXIL-SAME: (ptr {{.*}} sret(%"class.hlsl::RWBuffer") align 4 %[[Tmp0]],
  // DXIL-SAME: i32 noundef 10, i32 noundef 1, i32 noundef 4, i32 noundef 2, ptr noundef @[[BufA]])
  // SPV: call void @hlsl::RWBuffer<float>::__createFromBinding(unsigned int, unsigned int, int, unsigned int, char const*)
  // SPV-SAME: (ptr {{.*}} sret(%"class.hlsl::RWBuffer") align 8 %[[Tmp0]],
  // SPV-SAME: i32 noundef 12, i32 noundef 2, i32 noundef 4, i32 noundef 2, ptr noundef @[[BufA]])

  // Make sure B[3] is translated to a RWBuffer<int>::__createFromImplicitBinding call with range 5 and index 3
  // and DXIL for implicit binding in space0, order id 0
  // and SPIR-V explicit binding (binding 13, set 0)
  // DXIL: call void @hlsl::RWBuffer<int>::__createFromImplicitBinding(unsigned int, unsigned int, int, unsigned int, char const*)
  // DXIL-SAME: (ptr {{.*}} sret(%"class.hlsl::RWBuffer.0") align 4 %[[Tmp1]],
  // DXIL-SAME: i32 noundef 0, i32 noundef 0, i32 noundef 5, i32 noundef 3, ptr noundef @[[BufB]])
  // SPV: call void @hlsl::RWBuffer<int>::__createFromBinding(unsigned int, unsigned int, int, unsigned int, char const*)
  // SPV-SAME: (ptr {{.*}} sret(%"class.hlsl::RWBuffer.0") align 8 %[[Tmp1]],
  // SPV-SAME: i32 noundef 13, i32 noundef 0, i32 noundef 5, i32 noundef 3, ptr noundef @[[BufB]])

  // Make sure C[1] is translated to a RWBuffer<int>::__createFromBinding call with range 3 and index 1
  // and DXIL explicit binding (u2, space0)
  // and SPIR-V explicit binding (binding 2, set 0)
  // DXIL: call void @hlsl::RWBuffer<int>::__createFromBinding(unsigned int, unsigned int, int, unsigned int, char const*)
  // DXIL-SAME: (ptr {{.*}} sret(%"class.hlsl::RWBuffer.0") align 4 %[[Tmp2]],
  // DXIL-SAME: i32 noundef 2, i32 noundef 0, i32 noundef 3, i32 noundef 1, ptr noundef @[[BufC]])
  // SPV: call void @hlsl::RWBuffer<int>::__createFromBinding(unsigned int, unsigned int, int, unsigned int, char const*)
  // SPV-SAME: (ptr {{.*}} sret(%"class.hlsl::RWBuffer.0") align 8 %[[Tmp2]],
  // SPV-SAME: i32 noundef 2, i32 noundef 0, i32 noundef 3, i32 noundef 1, ptr noundef @[[BufC]])

  // Make sure D[7] is translated to a RWBuffer<double>::__createFromImplicitBinding call
  // for both DXIL and SPIR-V
  // DXIL: call void @hlsl::RWBuffer<double>::__createFromImplicitBinding(unsigned int, unsigned int, int, unsigned int, char const*)
  // DXIL-SAME: (ptr {{.*}} sret(%"class.hlsl::RWBuffer.1") align 4 %[[Tmp3]],
  // DXIL-SAME: i32 noundef 1, i32 noundef 0, i32 noundef 10, i32 noundef 7, ptr noundef @D.str)
  // SPV: call void @hlsl::RWBuffer<double>::__createFromImplicitBinding(unsigned int, unsigned int, int, unsigned int, char const*)
  // SPV-SAME: (ptr {{.*}} sret(%"class.hlsl::RWBuffer.1") align 8 %[[Tmp3]],
  // SPV-SAME: i32 noundef 0, i32 noundef 0, i32 noundef 10, i32 noundef 7, ptr noundef @[[BufD]])

  // Make sure E[5][0] is translated to RWBuffer<uint>::__createFromImplicitBinding call 
  // for both DXIL and SPIR-V with specified space/set 2
  // DXIL: call void  @hlsl::RWBuffer<unsigned int>::__createFromImplicitBinding(unsigned int, unsigned int, int, unsigned int, char const*)
  // DXIL-SAME: (ptr {{.*}} sret(%"class.hlsl::RWBuffer.2") align 4 %[[Tmp4]],
  // DXIL-SAME: i32 noundef 2, i32 noundef 2, i32 noundef 15, i32 noundef 5, ptr noundef @[[BufE]])
  // SPV: call void @hlsl::RWBuffer<unsigned int>::__createFromImplicitBinding(unsigned int, unsigned int, int, unsigned int, char const*)
  // SPV-SAME: (ptr {{.*}} sret(%"class.hlsl::RWBuffer.2") align 8 %[[Tmp4]],
  // SPV-SAME: i32 noundef 1, i32 noundef 2, i32 noundef 15, i32 noundef 5, ptr noundef @[[BufE]])
  Out[0] = A[2][0] + (float)B[3][0] + (float)C[1][0] + (float)D[7][0] + (float)E[5][0];
}
