// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   spirv-unknown-vulkan-compute %s -fsyntax-only -verify

struct S; // expected-note {{forward declaration of 'S'}}

// expected-error@hlsl/hlsl_spirv.h:24 {{argument type 'S' is incomplete}}

typedef vk::SpirvOpaqueType</* OpTypeArray */ 28, S, vk::integral_constant<uint, 4>> ArrayOfS; // #1
// expected-note@#1 {{in instantiation of template type alias 'SpirvOpaqueType' requested here}}

[numthreads(1, 1, 1)]
void main() {
    ArrayOfS buffers;
}
