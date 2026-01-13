// RUN: %clang_cc1 -triple spirv-linux-vulkan-vertex -x hlsl -emit-llvm -finclude-default-header -disable-llvm-passes -o - %s | FileCheck %s

// This is almost the same as semantic.explicit-mix-builtin.hlsl, except this
// time we build a vertex shader. This means the SV_Position semantic output
// is also a BuiltIn, This means we can mix implicit and explicit location
// assignment.
struct S1 {
  float4 position : SV_Position;
  [[vk::location(3)]] float4 color : A;
};

// CHECK: @SV_Position0 = external hidden thread_local addrspace(7) externally_initialized constant <4 x float>, !spirv.Decorations ![[#MD_0:]]
// CHECK: @SV_Position = external hidden thread_local addrspace(8) global <4 x float>, !spirv.Decorations ![[#MD_2:]]
// CHECK: @A0 = external hidden thread_local addrspace(8) global <4 x float>, !spirv.Decorations ![[#MD_0]]

[shader("vertex")]
S1 main1(float4 position : SV_Position) {
  S1 output;
  output.position = position;
  output.color = position;
  return output;
}

// CHECK: ![[#MD_0]] = !{![[#MD_1:]]}
// CHECK: ![[#MD_1]] = !{i32 30, i32 0}
//                            |       `-> Location index
//                            `-> SPIR-V decoration 'Location'
// CHECK: ![[#MD_2]] = !{![[#MD_3:]]}
// CHECK: ![[#MD_3]] = !{i32 11, i32 0}
//                            |       `-> BuiltIn 'Position'
//                            `-> SPIR-V decoration 'BuiltIn'
