// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.0-compute -x hlsl -fsyntax-only -verify \
// RUN:   -DRESOURCE="RWByteAddressBuffer" -DREGISTER=u1 %s
// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.0-compute -x hlsl -fsyntax-only -verify \
// RUN:   -DRESOURCE="ByteAddressBuffer" -DREGISTER=t0 %s
// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.0-compute -x hlsl -fsyntax-only -verify \
// RUN:   "-DRESOURCE=StructuredBuffer<uint>" -DREGISTER=t0 %s
// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.0-compute -x hlsl -fsyntax-only -verify \
// RUN:   "-DRESOURCE=RWStructuredBuffer<uint>" -DREGISTER=u1 %s
// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.0-compute -x hlsl -fsyntax-only -verify \
// RUN:   "-DRESOURCE=Buffer<uint>" -DREGISTER=t0 %s
// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.0-compute -x hlsl -fsyntax-only -verify \
// RUN:   "-DRESOURCE=RWBuffer<uint>" -DREGISTER=u1 %s

// expected-no-diagnostics

// Regression test for https://github.com/llvm/llvm-project/issues/192557
// Load on a const resource (e.g. const RWByteAddressBuffer) should be callable
// because Load does not modify the buffer. DXC accepts this pattern.

RESOURCE gBuf : register(REGISTER);
RWByteAddressBuffer gOut : register(u0);

void UseConst(const RESOURCE buf, out uint val) {
    val = buf.Load(0);
}

void UseConstWithStatus(const RESOURCE buf, out uint val) {
    uint status;
    val = buf.Load(0, status);
    val += status;
}

[numthreads(1,1,1)]
void main() {
    const RESOURCE local = gBuf;
    uint val;
    UseConst(local, val);
    uint val2;
    UseConstWithStatus(local, val2);
    // Write the loaded values to a separate buffer so they aren't dead-code
    // eliminated.
    gOut.Store(0, val);
    gOut.Store(4, val2);
}
