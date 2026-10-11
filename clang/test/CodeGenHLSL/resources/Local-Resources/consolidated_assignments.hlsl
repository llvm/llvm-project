// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple dxil-pc-shadermodel6.6-compute %s -emit-llvm -O1 -Wno-unused-value -o - | FileCheck %s

// Consolidated coverage for local-resource assignment/initialization patterns
// that must fold to a single unique global binding.
//
// Using two distinct globals (u0/u1) lets us prove that only GBuf0's binding
// materializes — all other reassignments fold away.
//
// Arrays of resources are covered separately by array.hlsl.

RWByteAddressBuffer GBuf0 : register(u0);
RWByteAddressBuffer GBuf1 : register(u1);

/// Mixed payload: a resource member alongside a non-resource one, so we also
/// cover a resource field being tracked independently of surrounding data.
struct ResHolder {
    RWByteAddressBuffer Buf;
    uint Value;
};

// Function returning a resource by way of a local — exercised as an
// aggregate-init element and as a chained-call receiver (`GetBuf().Store(...)`).
RWByteAddressBuffer GetBuf() {
    RWByteAddressBuffer Local = GBuf0;
    return Local;
}

// `inout` parameter: the callee both reads and could write back the handle.
void DoStore(inout RWByteAddressBuffer Buf, uint Idx) {
    Buf.Store(Idx, 42);
}
void ForwardStore(RWByteAddressBuffer Buf, uint Idx) {
    DoStore(Buf, Idx);
}

// `out` parameter: the callee assigns a global into the caller's local.
void WriteThrough(out RWByteAddressBuffer Buf) {
    Buf = GBuf0;
}

[numthreads(1,1,1)]
void main(uint GI : SV_GroupIndex) {
  // CHECK-LABEL: define {{.*}}@main(

  /// Only GBuf0 (register u0, space 0) materializes; every other
  /// init/reassignment resolves back to it or is folded away.
  // CHECK: %[[H:[^ ]+]] = tail call {{.*}}handlefrombinding{{.*}}(i32 0, i32 0,

  /// GBuf1 (u1) never materializes — all reassignments through it are folded
  /// away. Resources are materialized at the top of the entry point, so this
  /// must be checked here rather than at the end of the file.
  // CHECK-NOT: handlefrombinding{{.*}}(i32 0, i32 1,

  /// Aggregate init: brace-init a struct with a function-returned resource.
  ResHolder H = {GetBuf(), 0};
  /// Expression init: ternary in the initializer folds to H.Buf (= GBuf0).
  RWByteAddressBuffer Buf = (true ? H.Buf : GBuf1);
  /// Dead reassignment on the `false` path — folded away.
  if (false)
    Buf = GBuf1;
  /// Self-assign: `Buf = Buf` is a no-op.
  if (true)
    Buf = Buf;
  /// Alias chain: Buf → Alias preserves GBuf0's identity.
  RWByteAddressBuffer Alias = Buf;
  /// Comma initializer: right operand (= Alias = GBuf0) wins.
  RWByteAddressBuffer Comma = (GBuf1, Alias);

  /// Forwarding through a call chain: ForwardStore → DoStore (`inout`) stores
  /// 42 at offset 0.
  // CHECK: %[[P0:[^ ]+]] = call ptr {{.*}}getpointer{{.*}}(target({{.*}}) %[[H]], i32 0)
  // CHECK: store i32 42, ptr %[[P0]]
  ForwardStore(Comma, 0);

  /// Chained call: method invoked directly on a function return; stores at
  /// offset 4.
  // CHECK: %[[P1:[^ ]+]] = call ptr {{.*}}getpointer{{.*}}(target({{.*}}) %[[H]], i32 4)
  // CHECK: store i32 42, ptr %[[P1]]
  GetBuf().Store(4, 42);

  /// Init directly from a function return.
  // CHECK: %[[P2:[^ ]+]] = call ptr {{.*}}getpointer{{.*}}(target({{.*}}) %[[H]], i32 8)
  // CHECK: store i32 42, ptr %[[P2]]
  RWByteAddressBuffer FromRet = GetBuf();
  FromRet.Store(8, 42);

  /// Ternary whose arms are the same global — folds even under a runtime
  /// condition.
  // CHECK: %[[P3:[^ ]+]] = call ptr {{.*}}getpointer{{.*}}(target({{.*}}) %[[H]], i32 12)
  // CHECK: store i32 42, ptr %[[P3]]
  RWByteAddressBuffer BothSame = (GI != 0) ? GBuf0 : GBuf0;
  BothSame.Store(12, 42);

  /// `out` param: the callee initializes an otherwise-uninitialized local.
  // CHECK: %[[P4:[^ ]+]] = call ptr {{.*}}getpointer{{.*}}(target({{.*}}) %[[H]], i32 16)
  // CHECK: store i32 42, ptr %[[P4]]
  RWByteAddressBuffer FromOut;
  WriteThrough(FromOut);
  FromOut.Store(16, 42);

  /// Struct member assignment with a non-resource field alongside it.
  // CHECK: %[[P5:[^ ]+]] = call ptr {{.*}}getpointer{{.*}}(target({{.*}}) %[[H]], i32 20)
  // CHECK: store i32 42, ptr %[[P5]]
  ResHolder Mixed;
  Mixed.Buf = GBuf0;
  Mixed.Value = 42;
  Mixed.Buf.Store(20, Mixed.Value);
}
