// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   spirv-unknown-vulkan-compute %s -emit-llvm -disable-llvm-passes \
// RUN:   -o - | FileCheck %s

using Int = vk::SpirvType</* OpTypeInt */ 21, 4, 64, vk::Literal<vk::integral_constant<uint, 8>>, vk::Literal<vk::integral_constant<bool, false>>>;

// CHECK: %struct.S = type <{ i32, target("spirv.Type", target("spirv.Literal", 8), target("spirv.Literal", 0), 21, 4, 64), [4 x i8] }>
struct S {
    int a;
    Int b;
};

[numthreads(1,1,1)]
void main() {
    S value;
}
