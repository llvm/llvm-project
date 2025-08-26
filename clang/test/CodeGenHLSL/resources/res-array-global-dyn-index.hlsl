// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.6-compute -finclude-default-header \
// RUN:   -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s

// CHECK: @[[BufA:.*]] = private unnamed_addr constant [2 x i8] c"A\00", align 1

RWBuffer<float> A[4][3] : register(u2);
RWStructuredBuffer<float> Out;

// Make sure A[GI.x][GI.y] is translated to a RWBuffer<float> constructor call with range 12 and dynamically calculated index

// NOTE:
// Constructor call for explicit binding has "jjij" in the mangled name and the arguments are (register, space, range_size, index, name).

// CHECK: define internal void @_Z4mainDv3_j(<3 x i32> noundef %GI)
// CHECK: %[[GI_alloca:.*]] = alloca <3 x i32>, align 16
// CHECK: %[[Tmp0:.*]] = alloca %"class.hlsl::RWBuffer
// CHECK: store <3 x i32> %GI, ptr %[[GI_alloca]]

// CHECK: %[[GI:.*]] = load <3 x i32>, ptr %[[GI_alloca]], align 16
// CHECK: %[[GI_y:.*]] = extractelement <3 x i32> %[[GI]], i32 1
// CHECK: %[[GI:.*]] = load <3 x i32>, ptr %[[GI_alloca]], align 16
// CHECK: %[[GI_x:.*]] = extractelement <3 x i32> %[[GI]], i32 0
// CHECK: %[[Tmp1:.*]] = mul i32 %[[GI_x]], 3
// CHECK: %[[Index:.*]] = add i32 %[[GI_y]], %[[Tmp1]]
// CHECK: call void @_ZN4hlsl8RWBufferIfEC1EjjijPKc(ptr {{.*}} %[[Tmp0]], i32 noundef 2, i32 noundef 0, i32 noundef 12, i32 noundef %[[Index]], ptr noundef @A.str)
[numthreads(4,1,1)]
void main(uint3 GI : SV_GroupThreadID) {
  Out[0] = A[GI.x][GI.y][0];
}
