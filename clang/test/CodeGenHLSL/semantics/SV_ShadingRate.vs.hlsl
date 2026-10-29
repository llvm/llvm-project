// RUN: %clang_cc1 -triple spirv-unknown-vulkan1.3-vertex -x hlsl -emit-llvm -finclude-default-header -disable-llvm-passes -o - %s | FileCheck %s --check-prefix=CHECK-SPIRV
// RUN: %clang_cc1 -triple dxil-unknown-shadermodel6.4-vertex -x hlsl -emit-llvm -finclude-default-header -disable-llvm-passes -o - %s | FileCheck %s --check-prefix=CHECK-DXIL

// Behavior parity with DXC's hlsl/semantics/sv_shadingrate/shadingrate2.hlsl.

struct VS_OUT {
  float4 pos : SV_Position;
  uint rate : SV_ShadingRate;
};

// CHECK-SPIRV-DAG: @SV_ShadingRate = external hidden thread_local addrspace(8) global i32, !spirv.Decorations ![[#R_0:]]

// CHECK-SPIRV: define void @main() {{.*}} {
// CHECK-DXIL:  define void @main() {{.*}} {
VS_OUT main(float4 pos : Pos, uint rate : ShadingRate) {
  VS_OUT Out;
  Out.pos = pos;
  Out.rate = rate;
  return Out;
}

// SPIR-V: the user input `rate : ShadingRate` is a plain location input and
// only the SV_ShadingRate output is a built-in.
// CHECK-SPIRV: store i32 %{{.*}}, ptr addrspace(8) @SV_ShadingRate, align 4

// CHECK-DXIL: call void @llvm.dx.store.output.v4f32(i32 0, i32 0, i8 0, <4 x float> %{{.*}})
// CHECK-DXIL: call void @llvm.dx.store.output.i32(i32 1, i32 0, i8 0, i32 %{{.*}})

// A single decoration node: BuiltIn PrimitiveShadingRateKHR (4432), no Flat.
// CHECK-SPIRV-DAG: ![[#R_0]] = !{![[#R_1:]]}
// CHECK-SPIRV-DAG: ![[#R_1]] = !{i32 11, i32 4432}
