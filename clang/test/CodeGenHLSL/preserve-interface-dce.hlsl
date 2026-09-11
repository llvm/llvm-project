// RUN: %clang_cc1 -triple spirv-unknown-vulkan1.3-vertex -std=hlsl2021 \
// RUN:   -finclude-default-header -disable-llvm-passes -emit-llvm -o - %s \
// RUN:   | FileCheck %s --check-prefix=O0
// RUN: %clang_cc1 -triple spirv-unknown-vulkan1.3-vertex -std=hlsl2021 \
// RUN:   -finclude-default-header -O2 -emit-llvm -o - %s \
// RUN:   | FileCheck %s --check-prefix=O2
//
// Confirm that the frontend creates addrspace(7) globals for all entry-point
// input parameters (O0), and that GlobalDCE removes the unused one at O2.
// A passing O2 run confirms that -fspv-preserve-interface requires a DCE guard.

// Both input globals must be present before optimization.
// O0-DAG: @POSITION0 = external hidden thread_local addrspace(7)
// O0-DAG: @TEXCOORD0 = external hidden thread_local addrspace(7)

// The used POSITION0 global must survive optimization.
// O2: @POSITION0 = external hidden thread_local{{.*}}addrspace(7)

// The unused TEXCOORD0 global must be eliminated by GlobalDCE at -O2.
// O2-NOT: @TEXCOORD0 = {{.*}}addrspace(7)

[shader("vertex")]
float4 main(float4 pos : POSITION0, float4 uv : TEXCOORD0) : SV_Position {
  return pos;
}
