// RUN: %clang_cc1 -triple spirv-pc-vulkan-compute -x hlsl -emit-llvm -finclude-default-header -disable-llvm-passes -o - %s | FileCheck %s

[[vk::push_constant]]
struct {
    int    a;
    float  b;
    float3 c;
}
PushConstants;

// CHECK: %struct.anon = type <{ i32, float, <3 x float> }>
// CHECK: @PushConstants = external hidden addrspace(13) externally_initialized global %struct.anon, align 1

[numthreads(1, 1, 1)]
void main() {
  float tmp = PushConstants.b;
}
