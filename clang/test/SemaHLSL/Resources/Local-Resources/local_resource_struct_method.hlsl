// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.3-compute %s -emit-llvm -o - -verify

// expected-no-diagnostics

// Test that a user-defined struct with a member function can use a
// resource member. This exercises method dispatch through user-defined
// struct methods operating on resource handles.
//
// DXC: passes (both sema and codegen).

RWByteAddressBuffer gBuf0 : register(u0);

struct Wrapper {
    RWByteAddressBuffer buf;
    void DoStore(uint idx, uint val) { buf.Store(idx * 4, val); }
};

[numthreads(1,1,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    Wrapper w;
    w.buf = gBuf0;
    w.DoStore(tid.x, 42);
}
