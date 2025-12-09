// RUN: %clang_cc1 -triple spirv-pc-vulkan1.3-pixel -x hlsl -emit-llvm -finclude-default-header -disable-llvm-passes -o - %s | FileCheck %s --check-prefix=CHECK-SPIRV
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-pixel -x hlsl -emit-llvm -finclude-default-header -disable-llvm-passes -o - %s | FileCheck %s --check-prefix=CHECK-DXIL

// CHECK-SPIRV: @SV_Position = external hidden thread_local addrspace(7) externally_initialized constant <4 x float>, !spirv.Decorations ![[#MD_0:]]

// CHECK: define void @main() {{.*}} {
float4 main(float4 p : SV_Position) : A {
  // CHECK-SPIRV: %[[#P:]] = load <4 x float>, ptr addrspace(7) @SV_Position, align 16
  // CHECK-SPIRV: %[[#R:]] = call spir_func <4 x float> @_Z4mainDv4_f(<4 x float> %[[#P]])
  // CHECK-SPIRV:            store <4 x float> %[[#R]], ptr addrspace(8) @A0, align 16

  // CHECK-DXIL: %SV_Position0 = call <4 x float> @llvm.dx.load.input.v4f32(i32 4, i32 0, i32 0, i8 0, i32 poison)
  // CHECK-DXIL:    %[[#TMP:]] = call <4 x float> @_Z4mainDv4_f(<4 x float> %SV_Position0)
  // CHECK-DXIL:                 call void @llvm.dx.store.output.v4f32(i32 4, i32 0, i32 0, i8 0, i32 poison, <4 x float> %[[#TMP]])
  return p;
}

// CHECK-SPIRV-DAG: ![[#MD_0]] = !{![[#MD_1:]]}
// CHECK-SPIRV-DAG: ![[#MD_1]] = !{i32 11, i32 15}
//                                      |       `-> BuiltIn Position
//                                      `-> SPIR-V decoration 'FragCoord'
