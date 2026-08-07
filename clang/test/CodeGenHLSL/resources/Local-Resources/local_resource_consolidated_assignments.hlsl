// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple dxil-pc-shadermodel6.6-compute %s -emit-llvm -O1 -Wno-hlsl-explicit-binding -Wno-unused-value -o - | FileCheck %s

// Consolidated coverage for local-resource assignment/initialization patterns
// that must fold to a single unique global binding.
//
// Using two distinct globals (u0/u1) lets us prove that only GBuf0's binding
// materializes — all other reassignments fold away.
//
// Array assignment and array copy are covered separately by
// local_resource_array.hlsl and local_resource_array_copy.hlsl.

RWByteAddressBuffer GBuf0 : register(u0);
RWByteAddressBuffer GBuf1 : register(u1);

struct ResHolder { RWByteAddressBuffer Buf; };

// Function returning a resource — exercised as an aggregate-init element and as a
// chained-call receiver (`GetBuf().Store(...)`).
RWByteAddressBuffer GetBuf() { return GBuf0; }

void DoStore(RWByteAddressBuffer Buf, uint Idx) {
    Buf.Store(Idx, 42);
}
void ForwardStore(RWByteAddressBuffer Buf, uint Idx) {
    DoStore(Buf, Idx);
}

[numthreads(1,1,1)]
void main(uint GI : SV_GroupIndex) {
  // Aggregate init: brace-init a struct with a function-returned resource.
  ResHolder H = {GetBuf()};
  // Expression init: ternary in the initializer folds to H.Buf (= GBuf0).
  RWByteAddressBuffer Buf = (true ? H.Buf : GBuf1);
  // Dead reassignment on the `false` path — folded away.
  if (false)
    Buf = GBuf1;
  // Self-assign: `Buf = Buf` is a no-op.
  if (true)
    Buf = Buf;
  // Alias chain: Buf → Alias preserves GBuf0's identity.
  RWByteAddressBuffer Alias = Buf;
  // Comma initializer: right operand (= Alias = GBuf0) wins.
  RWByteAddressBuffer Comma = (GBuf1, Alias);
  // Forwarding through a call chain: ForwardStore → DoStore stores at offset 0.
  ForwardStore(Comma, 0);
  // Chained call: method invoked directly on a function return; stores at offset 4.
  GetBuf().Store(4, 42);
  // Init directly from a function return.
  RWByteAddressBuffer FromRet = GetBuf();
  FromRet.Store(8, 42);
  // Ternary whose arms are the same global — folds even under a runtime condition.
  RWByteAddressBuffer BothSame = (GI != 0) ? GBuf0 : GBuf0;
  BothSame.Store(12, 42);
}

// CHECK-LABEL: define {{.*}}@main(
// Only GBuf0 (register u0, space 0) materializes; every other init/reassignment
// resolves back to it or is folded away.
// CHECK: %[[H:[^ ]+]] = tail call {{.*}}handlefrombinding{{.*}}(i32 0, i32 0,
// ForwardStore → DoStore: store 42 at offset 0 via GBuf0's handle.
// CHECK: %[[P0:[^ ]+]] = call ptr {{.*}}getpointer{{.*}}(target({{.*}}) %[[H]], i32 0)
// CHECK: store i32 42, ptr %[[P0]]
// Chained call: GetBuf().Store(4, 42) also uses GBuf0's handle.
// CHECK: %[[P1:[^ ]+]] = call ptr {{.*}}getpointer{{.*}}(target({{.*}}) %[[H]], i32 4)
// CHECK: store i32 42, ptr %[[P1]]
// Init from a function return: stores at offset 8 via the same handle.
// CHECK: %[[P2:[^ ]+]] = call ptr {{.*}}getpointer{{.*}}(target({{.*}}) %[[H]], i32 8)
// CHECK: store i32 42, ptr %[[P2]]
// Same-arm ternary: folds to a single handle with no branch, stores at offset 12.
// CHECK: %[[P3:[^ ]+]] = call ptr {{.*}}getpointer{{.*}}(target({{.*}}) %[[H]], i32 12)
// CHECK: store i32 42, ptr %[[P3]]
// GBuf1 (u1) never materializes — all reassignments through it are folded away.
// CHECK-NOT: handlefrombinding{{.*}}(i32 0, i32 1,
