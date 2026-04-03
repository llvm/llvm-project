// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.3-compute %s -emit-llvm -o - -verify

// Test that using a resource type as an element of RWStructuredBuffer
// is rejected because resource types are intangible.
//
// DXC: also fails sema with a different message:
//   "object 'RWByteAddressBuffer' is not allowed in builtin template parameters"

RWByteAddressBuffer gBuf0 : register(u0);

void Fail_LocalBuffer() {
    RWStructuredBuffer<RWByteAddressBuffer> badBuffer;
    // expected-error@-1 {{constraints not satisfied for class template 'RWStructuredBuffer' [with element_type = RWByteAddressBuffer]}}
    // expected-note@*:* {{because 'RWByteAddressBuffer' does not satisfy '__is_structured_resource_element_compatible'}}
    // expected-note@*:* {{because '!__builtin_hlsl_is_intangible(hlsl::RWByteAddressBuffer)' evaluated to false}}
}

[numthreads(1,1,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    Fail_LocalBuffer();
}
