// RUN: %clang_cc1 -triple spirv-pc-vulkan-compute -x hlsl -emit-llvm -finclude-default-header -disable-llvm-passes -o - %s | FileCheck %s

struct T {
                 float2   f1[3];
    // FIXME(): matrix support.
    // column_major float3x2 f2[2];
    // row_major    int3x2   f4[2];
    // row_major    float3x2 f3[2];
};
// %struct.T = type { [3 x <2 x float>] }

struct S {
              float    f1;
              float3   f2;
              T        f4;
    // FIXME(): matrix support.
    // row_major int2x3   f5;
    // row_major float2x3 f3;
};
// %struct.S = type <{ float, <3 x float>, %struct.T }>

[[vk::push_constant]]
S pcs;
// CHECK: @pcs = external hidden addrspace(13) externally_initialized global %struct.S, align 1

[numthreads(1, 1, 1)]
void main() {
  float a = pcs.f1;
// CHECK: %[[#TMP:]] = load float, ptr addrspace(13) @pcs, align 1
// CHECK:              store float %[[#TMP]], ptr %a, align 4
}
