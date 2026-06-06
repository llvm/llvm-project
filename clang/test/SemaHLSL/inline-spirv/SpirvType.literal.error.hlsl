// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   spirv-unknown-vulkan-compute %s -fsyntax-only -verify

// expected-error@hlsl/hlsl_spirv.h:18 {{the argument to vk::Literal must be a vk::integral_constant}}

typedef vk::SpirvOpaqueType<28, vk::Literal<float>> T; // #1
// expected-note@#1 {{in instantiation of template type alias 'SpirvOpaqueType' requested here}}

[numthreads(1, 1, 1)]
void main() {
}
