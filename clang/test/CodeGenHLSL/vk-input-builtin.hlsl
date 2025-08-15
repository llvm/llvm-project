// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   spirv-unknown-vulkan1.3-compute %s -emit-llvm -O3 -o - | FileCheck %s

[[vk::ext_builtin_input(/* WorkgroupId */ 26)]]
static const uint3 groupid;
// CHECK: @_ZL7groupid = external hidden local_unnamed_addr addrspace(7) externally_initialized constant <3 x i32>, align 16, !spirv.Decorations [[META0:![0-9]+]]

RWStructuredBuffer<int> output : register(u1, space0);

[numthreads(1, 1, 1)]
void main() {
  output[0] = groupid;
}
// CHECK: [[META0]] = !{[[META1:![0-9]+]]}
// CHECK: [[META1]] = !{i32 11, i32 26}
