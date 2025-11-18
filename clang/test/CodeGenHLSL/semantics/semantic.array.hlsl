// RUN: %clang_cc1 -triple spirv-linux-vulkan-library -x hlsl -emit-llvm -finclude-default-header -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK-SPIRV -DTARGET=spv
// RUN: %clang_cc1 -triple dxil-px-shadermodel6.3-library -x hlsl -emit-llvm -finclude-default-header -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK-DXIL -DTARGET=dx

struct S0 {
  float4 position[2];
  float4 color;
};

// CHECK: %struct.S0 = type { [2 x <4 x float>], <4 x float> }

// CHECK-SPIRV: @A0 = external hidden thread_local addrspace(7) externally_initialized constant [2 x <4 x float>], !spirv.Decorations ![[#MD_0:]]
// CHECK-SPIRV: @A2 = external hidden thread_local addrspace(7) externally_initialized constant <4 x float>, !spirv.Decorations ![[#MD_2:]]

// CHECK:       define void @main0()
// CHECK-DXIL:          %A0 = call [2 x <4 x float>] @llvm.dx.load.input.a2v4f32(i32 4, i32 0, i32 0, i8 0, i32 poison)
// CHECK-DXIL:  %[[#TMP0:]] = insertvalue %struct.S0 poison, [2 x <4 x float>] %A0, 0
// CHECK-DXIL:          %A2 = call <4 x float> @llvm.dx.load.input.v4f32(i32 4, i32 0, i32 0, i8 0, i32 poison)
// CHECK-DXIL:  %[[#TMP1:]] = insertvalue %struct.S0 %[[#TMP0]], <4 x float> %A2, 1

// CHECK-SPIRV:   %[[#A0:]] = load [2 x <4 x float>], ptr addrspace(7) @A0, align 16
// CHECK-SPIRV: %[[#TMP0:]] = insertvalue %struct.S0 poison, [2 x <4 x float>] %[[#A0]], 0
// CHECK-SPIRV:  %[[#A01:]] = load <4 x float>, ptr addrspace(7) @A2, align 16
// CHECK-SPIRV: %[[#TMP1:]] = insertvalue %struct.S0 %[[#TMP0]], <4 x float> %[[#A01]], 1

// CHECK:        %[[#ARG:]] = alloca %struct.S0, align 16
// CHECK:                     store %struct.S0 %[[#TMP1]], ptr %[[#ARG]], align 16
// CHECK-DXIL:                call void @{{.*}}main0{{.*}}(ptr %[[#ARG]])
// CHECK-SPIRV:               call spir_func void @{{.*}}main0{{.*}}(ptr %[[#ARG]])
[shader("pixel")]
void main0(S0 p : A) {
  float tmp = p.position[0] + p.position[1] + p.color;
}

// CHECK-SPIRV: ![[#MD_0]] = !{![[#MD_1:]]}
// CHECK-SPIRV: ![[#MD_1]] = !{i32 30, i32 0}
// CHECK-SPIRV: ![[#MD_2]] = !{![[#MD_3:]]}
// CHECK-SPIRV: ![[#MD_3]] = !{i32 30, i32 2}
