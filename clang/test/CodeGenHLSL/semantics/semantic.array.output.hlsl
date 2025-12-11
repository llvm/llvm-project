// RUN: %clang_cc1 -triple spirv-linux-vulkan-library -x hlsl -emit-llvm -finclude-default-header -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK-SPIRV -DTARGET=spv
// RUN: %clang_cc1 -triple dxil-px-shadermodel6.3-library -x hlsl -emit-llvm -finclude-default-header -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK-DXIL -DTARGET=dx

struct S0 {
  float4 position[2];
  float4 color;
};

// CHECK-SPIRV-DAG:    @A0 = external hidden thread_local addrspace(7) externally_initialized constant <4 x float>, !spirv.Decorations ![[#METADATA_0:]]

[shader("pixel")]
S0 main1(float4 input : A) : B {
// CHECK:         %[[#ARG:]] = alloca %struct.S0, align 16
// CHECK-SPIRV: %[[#INPUT:]] = load <4 x float>, ptr addrspace(7) @A0, align 16
// CHECK-DXIL:           %A0 = call <4 x float> @llvm.dx.load.input.v4f32(i32 4, i32 0, i32 0, i8 0, i32 poison)
// CHECK-DXIL:                 call void @{{.*}}main1{{.*}}(ptr %[[#ARG]], <4 x float> %A0)
// CHECK-SPIRV:                call spir_func void @{{.*}}main1{{.*}}(ptr %[[#ARG]], <4 x float> %[[#INPUT]])

  // CHECK:        %[[#ST:]] = load %struct.S0, ptr %[[#ARG]], align 16
  // CHECK:       %[[#TMP:]] = extractvalue %struct.S0 %[[#ST]], 0
  // CHECK-SPIRV:              store [2 x <4 x float>] %[[#TMP]], ptr addrspace(8) @B0, align 16
  // CHECK-DXIL:               call void @llvm.dx.store.output.a2v4f32(i32 4, i32 0, i32 0, i8 0, i32 poison, [2 x <4 x float>] %[[#TMP]])
  // CHECK:       %[[#TMP:]] = extractvalue %struct.S0 %[[#ST]], 1
  // CHECK-SPIRV:              store <4 x float> %[[#TMP]], ptr addrspace(8) @B2, align 16
  // CHECK-DXIL:               call void @llvm.dx.store.output.v4f32(i32 4, i32 0, i32 0, i8 0, i32 poison, <4 x float> %[[#TMP]])

  S0 output;
  output.position[0] = input;
  output.position[1] = input;
  output.color = input;
  return output;
}

// CHECK-SPIRV-DAG: ![[#METADATA_0]] = !{![[#METADATA_1:]]}
// CHECK-SPIRV-DAG: ![[#METADATA_1]] = !{i32 30, i32 0}
//                                            |      `- Location index
//                                            `-> Decoration "Location"
