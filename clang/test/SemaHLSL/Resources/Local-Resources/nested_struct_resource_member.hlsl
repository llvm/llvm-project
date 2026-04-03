// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-compute %s -emit-llvm -o - -verify

// expected-no-diagnostics

// DXC: passes (both sema and codegen).

RWByteAddressBuffer gBuf1 : register(u1);

struct NestedInner { RWByteAddressBuffer buf; };
struct NestedOuter { NestedInner inner; };

uint Pass_NestedStruct(uint idx) {
    NestedOuter s;
    s.inner.buf = gBuf1;
    s.inner.buf.Store(idx * 4, 28);

    return 28;
}

[numthreads(8,8,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x + tid.y * 8;
    Pass_NestedStruct(idx);
}
