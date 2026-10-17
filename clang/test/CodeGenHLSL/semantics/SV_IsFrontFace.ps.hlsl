// RUN: %clang_cc1 -triple spirv-pc-vulkan1.3-pixel -x hlsl -emit-llvm -finclude-default-header -disable-llvm-passes -o - %s | FileCheck %s --check-prefix=CHECK-SPIRV
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-pixel -x hlsl -emit-llvm -finclude-default-header -disable-llvm-passes -o - %s | FileCheck %s --check-prefix=CHECK-DXIL

// CHECK-SPIRV: @SV_IsFrontFace = external hidden thread_local addrspace(7) externally_initialized constant i1, !spirv.Decorations ![[#MD_0:]]

// CHECK-SPIRV: define void @main() {{.*}} {
// CHECK-DXIL:  define void @main() {{.*}} {
float4 main(bool ff : SV_IsFrontFace) : SV_Target {
  // CHECK-SPIRV: %[[FF:.*]] = load i1, ptr addrspace(7) @SV_IsFrontFace
  // CHECK-SPIRV: %[[R:.*]] = call spir_func <4 x float> @_Z4mainb(i1 %[[FF]])

  // CHECK-DXIL: %[[INPUT:.*]] = call i32 @llvm.dx.load.input.i32(i32 0, i32 0, i8 0, i32 poison)
  // CHECK-DXIL: %[[BOOL:.*]] = icmp ne i32 %[[INPUT]], 0
  // CHECK-DXIL: %[[R:.*]] = call <4 x float> @_Z4mainb(i1 %[[BOOL]])
  return ff ? float4(1, 0, 0, 1) : float4(0, 0, 1, 1);
}

// Two decorations in one node: BuiltIn FrontFacing (17) and Flat (14).
// CHECK-SPIRV-DAG: ![[#MD_0]] = !{![[#MD_1:]], ![[#MD_2:]]}
// CHECK-SPIRV-DAG: ![[#MD_1]] = !{i32 11, i32 17}
// CHECK-SPIRV-DAG: ![[#MD_2]] = !{i32 14}
