// RUN: %clang_cc1 -triple dxil-unknown-shadermodel6.8-vertex -x hlsl -emit-llvm -finclude-default-header -disable-llvm-passes -o - %s | FileCheck --check-prefix=CHECK-DXIL %s
// RUN: %clang_cc1 -triple spirv-unknown-vulkan1.3-vertex -x hlsl -emit-llvm -finclude-default-header -disable-llvm-passes -o - %s | FileCheck --check-prefix=CHECK-SPIRV  %s

// CHECK-SPIRV: @SV_Position0 = external hidden thread_local addrspace(7) externally_initialized constant <4 x float>, !spirv.Decorations ![[#MD_0:]]
// CHECK-SPIRV: @SV_Position = external hidden thread_local addrspace(8) global <4 x float>, !spirv.Decorations ![[#MD_2:]]

// CHECK: define void @main() {{.*}} {
float4 main(float4 p : SV_Position) : SV_Position {
  // CHECK-SPIRV: %[[#P:]] = load <4 x float>, ptr addrspace(7) @SV_Position0, align 16
  // CHECK-SPIRV: %[[#R:]] = call spir_func <4 x float> @_Z4mainDv4_f(<4 x float> %[[#P]])
  // CHECK-SPIRV:            store <4 x float> %[[#R]], ptr addrspace(8) @SV_Position, align 16

  // CHECK-DXIL: %SV_Position0 = call <4 x float> @llvm.dx.load.input.v4f32(i32 4, i32 0, i32 0, i8 0, i32 poison)
  // CHECK-DXIL:    %[[#TMP:]] = call <4 x float> @_Z4mainDv4_f(<4 x float> %SV_Position0)
  // CHECK-DXIL:                 call void @llvm.dx.store.output.v4f32(i32 4, i32 0, i32 0, i8 0, i32 poison, <4 x float> %[[#TMP]])
  return p;
}

// CHECK-SPIRV-DAG: ![[#MD_0]] = !{![[#MD_1:]]}
// CHECK-SPIRV-DAG: ![[#MD_2]] = !{![[#MD_3:]]}
// CHECK-SPIRV-DAG: ![[#MD_1]] = !{i32 30, i32 0}
//                                      |       `-> Location 0
//                                      `-> SPIR-V decoration 'Location'
// CHECK-SPIRV-DAG: ![[#MD_3]] = !{i32 11, i32 0}
//                                      |       `-> BuiltIn Position
//                                      `-> SPIR-V decoration 'BuiltIn'
