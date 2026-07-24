// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-compute %s -emit-llvm -o - -verify

// expected-note@*:* {{because 'RWByteAddressBuffer' does not satisfy '__is_structured_resource_element_compatible'}}
// expected-note@*:* {{because '!__builtin_hlsl_is_intangible(hlsl::RWByteAddressBuffer)' evaluated to false}}
RWByteAddressBuffer GBuf0 : register(u0);

void Fail_LocalBuffer() {
// expected-error@+1 {{constraints not satisfied for class template 'RWStructuredBuffer' [with element_type = RWByteAddressBuffer]}}
    RWStructuredBuffer<RWByteAddressBuffer> BadBuffer;
}

[numthreads(1,1,1)]
void main(uint3 Tid : SV_DispatchThreadID) {
    Fail_LocalBuffer();
}
