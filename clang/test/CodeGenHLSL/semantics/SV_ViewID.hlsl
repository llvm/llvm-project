// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.1-vertex -x hlsl -emit-llvm -finclude-default-header -disable-llvm-passes -o - %s | FileCheck --check-prefix=CHECK-DXIL %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.1-pixel -x hlsl -emit-llvm -finclude-default-header -disable-llvm-passes -o - %s | FileCheck --check-prefix=CHECK-DXIL %s
// RUN: %clang_cc1 -triple spirv-pc-vulkan1.3-vertex -x hlsl -emit-llvm -finclude-default-header -disable-llvm-passes -o - %s | FileCheck --check-prefixes=CHECK-SPIRV,SPIRV-VS %s
// RUN: %clang_cc1 -triple spirv-pc-vulkan1.3-pixel -x hlsl -emit-llvm -finclude-default-header -disable-llvm-passes -o - %s | FileCheck --check-prefixes=CHECK-SPIRV,SPIRV-PS %s

#if __SHADER_TARGET_STAGE == __SHADER_STAGE_PIXEL
#define OUTPUT SV_Target
#else
#define OUTPUT SV_Position
#endif

// CHECK-SPIRV: @SV_ViewID = external hidden thread_local addrspace(7) externally_initialized constant i32, !spirv.Decorations ![[#MD_0:]]

// The view ID is read with a dedicated operation and is not part of the input
// signature, so the user input that follows keeps signature element 0.

// CHECK-DXIL-LABEL: define void @main()
// CHECK-DXIL: %[[VIEW:.*]] = call i32 @llvm.dx.view.id()
// CHECK-DXIL: %[[A:.*]] = call <4 x float> @llvm.dx.load.input.v4f32(i32 0, i32 0, i8 0, i32 poison)
// CHECK-DXIL: call <4 x float> @_Z4mainjDv4_f(i32 %[[VIEW]], <4 x float> %[[A]])
// CHECK-DXIL-NOT: !"SV_ViewID"

// CHECK-SPIRV-LABEL: define void @main()
// CHECK-SPIRV: %[[VIEW:.*]] = load i32, ptr addrspace(7) @SV_ViewID, align 4
// CHECK-SPIRV: call spir_func <4 x float> @_Z4mainjDv4_f(i32 %[[VIEW]],
float4 main(uint view : SV_ViewID, float4 a : A) : OUTPUT {
  return a * view;
}

// The pixel shader input is flat because it is an integer.
// SPIRV-VS-DAG: ![[#MD_0]] = !{![[#BUILTIN:]]}
// SPIRV-PS-DAG: ![[#MD_0]] = !{![[#BUILTIN:]], ![[#FLAT:]]}
// SPIRV-PS-DAG: ![[#FLAT]] = !{i32 14}
// The metadata below contains SPIR-V decoration 'BuiltIn' followed by
// BuiltIn ViewIndex.
// CHECK-SPIRV-DAG: ![[#BUILTIN]] = !{i32 11, i32 4440}
