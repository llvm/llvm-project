// RUN: %clang_cc1 -triple spirv-unknown-vulkan-compute -x hlsl -emit-llvm -disable-llvm-passes -o - -hlsl-entry main %s -verify

struct S {
    float f;
};

// expected-error@+1 {{'vk::binding' attribute is not compatible with 'vk::push_constant' attribute}}
[[vk::push_constant, vk::binding(5)]]
S pcs;

[numthreads(1, 1, 1)]
void main() {
}
