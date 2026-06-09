// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   spirv-unknown-vulkan1.3-vertex %s -emit-llvm -O3 -o - | FileCheck %s

[[vk::ext_builtin_output(/* Position */ 0)]]
static float4 position;
// CHECK: @position = external hidden local_unnamed_addr addrspace(8) global <4 x float>, align 4, !spirv.Decorations [[META0:![0-9]+]]

RWStructuredBuffer<float4> input : register(u1, space0);

void main() {
  position = input[0];
  // CHECK: store <4 x float> %[[#]], ptr addrspace(8) @position, align 4
}
// CHECK: [[META0]] = !{[[META1:![0-9]+]]}
// CHECK: [[META1]] = !{i32 11, i32 0}
