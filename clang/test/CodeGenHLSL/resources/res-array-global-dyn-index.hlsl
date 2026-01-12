// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.6-compute -finclude-default-header \
// RUN:   -emit-llvm -disable-llvm-passes -o - %s | llvm-cxxfilt | FileCheck %s

// CHECK: @[[BufA:.*]] = private unnamed_addr constant [2 x i8] c"A\00", align 1

RWBuffer<float> A[4][3] : register(u2);
RWStructuredBuffer<float> Out;

// Make sure A[GI.x][GI.y] is translated to a RWBuffer<float>::__createFromBinding call
// with range 12 and dynamically calculated index

// CHECK: define internal void @main(unsigned int vector[3])(<3 x i32> noundef %GI)
// CHECK: %[[GI_alloca:.*]] = alloca <3 x i32>, align 16
// CHECK: %[[Tmp0:.*]] = alloca %"class.hlsl::RWBuffer
// CHECK: store <3 x i32> %GI, ptr %[[GI_alloca]]

// CHECK: %[[GI:.*]] = load <3 x i32>, ptr %[[GI_alloca]], align 16
// CHECK: %[[GI_y:.*]] = extractelement <3 x i32> %[[GI]], i32 1
// CHECK: %[[GI:.*]] = load <3 x i32>, ptr %[[GI_alloca]], align 16
// CHECK: %[[GI_x:.*]] = extractelement <3 x i32> %[[GI]], i32 0
// CHECK: %[[Tmp1:.*]] = mul i32 %[[GI_x]], 3
// CHECK: %[[Index:.*]] = add i32 %[[GI_y]], %[[Tmp1]]
// CHECK: call void @hlsl::RWBuffer<float>::__createFromBinding(unsigned int, unsigned int, int, unsigned int, char const*)
// CHECK-SAME: (ptr {{.*}} sret(%"class.hlsl::RWBuffer") align 4 %[[Tmp0]],
// CHECK-SAME: i32 noundef 2, i32 noundef 0, i32 noundef 12, i32 noundef %[[Index]], ptr noundef @A.str)
[numthreads(4,1,1)]
void main(uint3 GI : SV_GroupThreadID) {
  Out[0] = A[GI.x][GI.y][0];
}
