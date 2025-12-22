// RUN: %clang_cc1 -triple spirv-pc-vulkan1.3-pixel -x hlsl -emit-llvm -finclude-default-header -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK-SPIRV
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-pixel -x hlsl -emit-llvm -finclude-default-header -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK-DXIL

// CHECK-SPIRV: @SV_Position = external hidden thread_local addrspace(7) externally_initialized constant <4 x float>, !spirv.Decorations ![[#MD_0:]]
// CHECK-SPIRV: @SV_Target0 = external hidden thread_local addrspace(8) global <4 x float>, !spirv.Decorations ![[#MD_2:]]

struct Output {
  [[vk::location(2)]] float4 field : SV_Target;
};

// CHECK: define void @main() {{.*}} {
Output main(float4 p : SV_Position) {
  // CHECK:   %[[#OUT:]] = alloca %struct.Output, align 16

  // CHECK-SPIRV:    %[[#IN:]] = load <4 x float>, ptr addrspace(7) @SV_Position, align 16
  // CHECK-SPIRV:                call spir_func void @_Z4mainDv4_f(ptr %[[#OUT]], <4 x float> %[[#IN]])

  // CHECK-DXIL:                 call void @_Z4mainDv4_f(ptr %[[#OUT]], <4 x float> %SV_Position0)

  // CHECK:   %[[#TMP:]] = load %struct.Output, ptr %[[#OUT]], align 16
  // CHECK: %[[#FIELD:]] = extractvalue %struct.Output %[[#TMP]], 0

  // CHECK-SPIRV:                store <4 x float> %[[#FIELD]], ptr addrspace(8) @SV_Target0, align 16
  // CHECK-DXIL:                 call void @llvm.dx.store.output.v4f32(i32 4, i32 0, i32 0, i8 0, i32 poison, <4 x float> %[[#FIELD]])
  Output o;
  o.field = p;
  return o;
}

// CHECK-SPIRV-DAG: ![[#MD_0]] = !{![[#MD_1:]]}
// CHECK-SPIRV-DAG: ![[#MD_1]] = !{i32 11, i32 15}
//                                      |       `-> BuiltIn 'FragCoord'
//                                      `-> SPIR-V decoration 'BuiltIn'
// CHECK-SPIRV-DAG: ![[#MD_2]] = !{![[#MD_3:]]}
// CHECK-SPIRV-DAG: ![[#MD_3]] = !{i32 30, i32 2}
//                                      |       `-> Location index
//                                      `-> SPIR-V decoration 'Location'
