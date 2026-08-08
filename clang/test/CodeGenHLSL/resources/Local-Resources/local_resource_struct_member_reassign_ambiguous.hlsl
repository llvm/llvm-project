// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple dxil-pc-shadermodel6.6-compute %s -emit-llvm -O1 -o - | FileCheck %s

// TODO: This test ought to produce an ambiguity warning, but currently does not.
// `H.Buf` is assigned GBuf0 and then conditionally reassigned GBuf1, so the
// resource it refers to at the Store is not statically known - the same pattern
// on a plain local variable is diagnosed with
// -Whlsl-explicit-binding ("... is not to the same unique global resource").
// Sema needs to close this gap and diagnose binding-ambiguous reassignment of
// resource-typed *struct members*, not just local resource variables. Once that
// diagnostic exists, this test should move to SemaHLSL and check for the warning.

RWByteAddressBuffer GBuf0 : register(u0);
RWByteAddressBuffer GBuf1 : register(u1);

struct ResHolder { RWByteAddressBuffer Buf; };

[numthreads(4,1,1)]
void main(uint Tid : SV_GroupThreadID) {
    ResHolder H;
    H.Buf = GBuf0;
    if (Tid)
      H.Buf = GBuf1;
    H.Buf.Store(0, 42);
}

// CHECK-LABEL: define {{.*}}@main(
// Binding for GBuf0 (register(u0, space0)) is emitted.
// CHECK-DAG: call {{.*}}handlefrombinding{{.*}}(i32 0, i32 0,
// Binding for GBuf1 (register(u1, space0)) is emitted.
// CHECK-DAG: call {{.*}}handlefrombinding{{.*}}(i32 0, i32 1,
