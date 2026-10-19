// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple dxil-pc-shadermodel6.6-compute %s -emit-llvm -o - -verify

// A *local* resource array indexed by a runtime value. Every element the index
// can select holds the same global, so there is one unique binding and nothing
// to warn about. `& 1` keeps the index off the unassigned elements 2 and 3.
// Sema-only: the backend can't lower this yet (llvm/llvm-project#192538).

// expected-no-diagnostics
RWByteAddressBuffer Out : register(u0);

[numthreads(1,1,1)]
void main(uint3 Tid : SV_DispatchThreadID) {
    RWByteAddressBuffer Arr[4];
    Arr[0] = Out;
    Arr[1] = Out;
    Arr[Tid.x & 1].Store(0, 42);
}
