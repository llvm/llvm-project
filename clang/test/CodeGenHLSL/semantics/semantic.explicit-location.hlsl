// RUN: %clang_cc1 -triple spirv-pc-vulkan1.3-pixel -x hlsl -emit-llvm -finclude-default-header -disable-llvm-passes -o - %s | FileCheck %s --check-prefix=CHECK-SPIRV
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-pixel -x hlsl -emit-llvm -finclude-default-header -disable-llvm-passes -o - %s | FileCheck %s --check-prefix=CHECK-DXIL

// CHECK-SPIRV: @SV_Target0 = external hidden thread_local addrspace(8) global <4 x float>, !spirv.Decorations ![[#MD_2:]]

// CHECK: define void @main() {{.*}} {
[[vk::location(2)]] float4 main(float4 p : SV_Position) : SV_Target {
  // CHECK-SPIRV: %[[RESULT:.*]] = call spir_func <4 x float> @_Z4mainDv4_f(<4 x float> %{{.*}})
  // CHECK-SPIRV:                 store <4 x float> %[[RESULT]], ptr addrspace(8) @SV_Target0, align 4

  // CHECK-DXIL: %[[INPUT:.*]] = call <4 x float> @llvm.dx.load.input.v4f32(i32 0, i32 0, i8 0, i32 poison)
  // CHECK-DXIL: %[[RESULT:.*]] = call <4 x float> @_Z4mainDv4_f(<4 x float> %[[INPUT]])
  // CHECK-DXIL:                 call void @llvm.dx.store.output.v4f32(i32 0, i32 0, i8 0, <4 x float> %[[RESULT]])
  return p;
}

// CHECK-SPIRV-DAG: ![[#MD_2]] = !{![[#MD_3:]]}
// CHECK-SPIRV-DAG: ![[#MD_3]] = !{i32 30, i32 2}
//                                      |       `-> Location index
//                                      `-> SPIR-V decoration 'Location'
