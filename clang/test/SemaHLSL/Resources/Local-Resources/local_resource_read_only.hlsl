// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.3-compute %s -emit-llvm -o - -verify

// expected-no-diagnostics

// Test that a read-only ByteAddressBuffer can be used as a local variable.
// This is distinct from RWByteAddressBuffer because it only supports Load
// (no Store), exercising different method resolution. Load is required here
// because ByteAddressBuffer is a read-only resource type.
//
// DXC: passes (both sema and codegen).

ByteAddressBuffer gBuf0 : register(t0);

uint Pass_ReadOnlyLocal(uint idx) {
    ByteAddressBuffer buf = gBuf0;
    return buf.Load(idx * 4);
}

[numthreads(1,1,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    Pass_ReadOnlyLocal(tid.x);
}
