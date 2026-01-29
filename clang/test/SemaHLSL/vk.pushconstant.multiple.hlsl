// RUN: %clang_cc1 -triple spirv-unknown-vulkan-compute -x hlsl -emit-llvm -disable-llvm-passes -o - -hlsl-entry main %s -verify

struct S {
    float f;
};

[[vk::push_constant]] S a;

// expected-error@+1 {{cannot have more than one push constant block}}
[[vk::push_constant]] S b;

[numthreads(1, 1, 1)]
void main() {}
