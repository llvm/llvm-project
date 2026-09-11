// RUN: %clang_cc1 -triple spirv-unknown-vulkan1.3-vertex -std=hlsl2021 \
// RUN:   -finclude-default-header -fspv-preserve-interface -O2 -emit-llvm \
// RUN:   -o - %s | FileCheck %s
//
// Confirm that -fspv-preserve-interface prevents GlobalDCE from removing an
// unused entry-point input semantic at -O2. Without the flag, @TEXCOORD0 is
// eliminated (see preserve-interface-dce.hlsl). With the flag, it must survive.

// Both input globals must be present in the optimized IR.
// CHECK-DAG: @POSITION0 = external hidden thread_local addrspace(7)
// CHECK-DAG: @TEXCOORD0 = external hidden thread_local addrspace(7)

// Both globals must appear in llvm.compiler.used. The implementation adds all
// addrspace(7) and addrspace(8) globals, not just the ones the optimizer would
// otherwise remove, matching DXC's behavior of marking the entire OpEntryPoint
// as live.
// CHECK: @llvm.compiler.used = appending
// CHECK-DAG: @POSITION0 to ptr
// CHECK-DAG: @TEXCOORD0 to ptr

[shader("vertex")]
float4 main(float4 pos : POSITION0, float4 uv : TEXCOORD0) : SV_Position {
  return pos;
}
