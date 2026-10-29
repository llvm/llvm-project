// RUN: %clang_cc1 -triple spirv-pc-vulkan1.3-pixel -x hlsl -emit-llvm -finclude-default-header -disable-llvm-passes -o - %s | FileCheck %s --check-prefix=CHECK-SPIRV
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.4-pixel -x hlsl -emit-llvm -finclude-default-header -disable-llvm-passes -o - %s | FileCheck %s --check-prefix=CHECK-DXIL

// Behavior parity with DXC's hlsl/semantics/sv_shadingrate/shadingrate1.hlsl.

// CHECK-SPIRV-DAG: @SV_ShadingRate = external hidden thread_local addrspace(7) externally_initialized constant i32, !spirv.Decorations ![[#MD_0:]]

// CHECK-SPIRV: define void @main() {{.*}} {
// CHECK-DXIL:  define void @main() {{.*}} {
float2 main(float2 v0 : Texcoord0, uint rate : SV_ShadingRate) : SV_Target {
  // CHECK-SPIRV: %[[RATE:.*]] = load i32, ptr addrspace(7) @SV_ShadingRate, align 4
  // CHECK-SPIRV: %[[R:.*]] = call spir_func <2 x float> @_Z4mainDv2_fj(<2 x float> %{{.*}}, i32 %[[RATE]])

  // CHECK-DXIL: %[[V0:.*]] = call <2 x float> @llvm.dx.load.input.v2f32(i32 0, i32 0, i8 0, i32 poison)
  // CHECK-DXIL: %[[RATE:.*]] = call i32 @llvm.dx.load.input.i32(i32 1, i32 0, i8 0, i32 poison)
  // CHECK-DXIL: %[[R:.*]] = call <2 x float> @_Z4mainDv2_fj(<2 x float> %[[V0]], i32 %[[RATE]])
  // CHECK-DXIL: call void @llvm.dx.store.output.v2f32(i32 0, i32 0, i8 0, <2 x float> %[[R]])
  uint2 rate2 = uint2(1 << (rate & 0x3), 1 << ((rate >> 2) & 0x3));
  return v0 * rate2;
}

// Both decorations live in a single node: BuiltIn ShadingRateKHR (4444) and
// Flat (14).
// CHECK-SPIRV-DAG: ![[#MD_0]] = !{![[#MD_1:]], ![[#MD_2:]]}
// CHECK-SPIRV-DAG: ![[#MD_1]] = !{i32 11, i32 4444}
// CHECK-SPIRV-DAG: ![[#MD_2]] = !{i32 14}
