// RUN: %clang_cc1 -triple spirv-unknown-vulkan1.3-pixel -x hlsl -emit-llvm -finclude-default-header -disable-llvm-passes -o - %s | FileCheck %s

// CHECK: @sv_position = external hidden thread_local addrspace(7) externally_initialized constant <4 x float>, !spirv.Decorations !0

// CHECK: define void @main() {{.*}} {
float4 main(float4 p : SV_Position) {
  // CHECK: %[[#P:]] = load <4 x float>, ptr addrspace(7) @sv_position, align 16
  // CHECK: %[[#R:]] = call spir_func <4 x float> @_Z4mainDv4_f(<4 x float> %[[#P]])
  return p;
}
