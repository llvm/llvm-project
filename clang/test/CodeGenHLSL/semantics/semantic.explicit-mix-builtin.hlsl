// RUN: %clang_cc1 -triple spirv-linux-vulkan-pixel -x hlsl -emit-llvm -finclude-default-header -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK-SPIRV

// The following code is allowed because the `SV_Position` semantic is here
// translated into a SPIR-V builtin. Meaning there is no implicit `Location`
// assignment.

struct S2 {
  float4 a;
  float4 b;
};

struct S1 {
  float4 position : SV_Position;
  [[vk::location(3)]] float4 color0 : COLOR0;
};

// CHECK-SPIRV: @SV_Position = external hidden thread_local addrspace(7) externally_initialized constant <4 x float>, !spirv.Decorations ![[#MD_0:]]
// CHECK-SPIRV: @COLOR0 = external hidden thread_local addrspace(7) externally_initialized constant <4 x float>, !spirv.Decorations ![[#MD_2:]]
// CHECK-SPIRV: @SV_Target0 = external hidden thread_local addrspace(8) global <4 x float>, !spirv.Decorations ![[#MD_4:]]

[shader("pixel")]
float4 main(S1 p) : SV_Target {
  return p.position + p.color0;
}
// CHECK-SPIRV:    %[[#SV_POS:]] = load <4 x float>, ptr addrspace(7) @SV_Position, align 16
// CHECK:            %[[#TMP1:]] = insertvalue %struct.S1 poison, <4 x float> %[[#SV_POS]], 0
// CHECK-SPIRV:        %[[#A0:]] = load <4 x float>, ptr addrspace(7) @COLOR0, align 16
// CHECK:            %[[#TMP2:]] = insertvalue %struct.S1 %[[#TMP1]], <4 x float> %[[#A0]], 1
// CHECK:               %[[#P:]] = alloca %struct.S1, align 16
// CHECK:                          store %struct.S1 %[[#TMP2]], ptr %[[#P]], align 16
// CHECK-SPIRV:         %[[#R:]] = call spir_func <4 x float> @_Z4main2S1(ptr %[[#P]])
// CHECK-SPIRV:                    store <4 x float> %[[#R]], ptr addrspace(8) @SV_Target0, align 16

// CHECK-SPIRV: ![[#MD_0]] = !{![[#MD_1:]]}
// CHECK-SPIRV: ![[#MD_1]] = !{i32 11, i32 15}
// CHECK-SPIRV: ![[#MD_2]] = !{![[#MD_3:]]}
// CHECK-SPIRV: ![[#MD_3]] = !{i32 30, i32 3}
// CHECK-SPIRV: ![[#MD_4]] = !{![[#MD_5:]]}
// CHECK-SPIRV: ![[#MD_5]] = !{i32 30, i32 0}
