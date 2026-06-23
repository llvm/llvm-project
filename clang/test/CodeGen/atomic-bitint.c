// RUN: %clang_cc1 -std=c23 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -std=c2y -triple x86_64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s
//
// Atomic operations on _BitInt(N). load/store/exchange/compare-exchange and
// bitwise RMW lower directly; arithmetic RMW on a padded width and any RMW on a
// width too wide for an inline atomicrmw lower to a compare-exchange loop that
// computes at the value width.

typedef _BitInt(37)          S37;
typedef unsigned _BitInt(37) U37;
typedef _BitInt(64)          S64;
typedef _BitInt(128)         S128;
typedef _BitInt(256)         S256;

// CHECK-LABEL: @ld37(
// CHECK: load atomic i64
S37 ld37(_Atomic(S37) *p) { return __c11_atomic_load(p, __ATOMIC_SEQ_CST); }

// CHECK-LABEL: @st37(
// CHECK: store atomic i64
void st37(_Atomic(S37) *p, S37 v) { __c11_atomic_store(p, v, __ATOMIC_SEQ_CST); }

// CHECK-LABEL: @xchg37(
// CHECK: atomicrmw xchg ptr {{.*}} i64
S37 xchg37(_Atomic(S37) *p, S37 v) {
  return __c11_atomic_exchange(p, v, __ATOMIC_SEQ_CST);
}

// CHECK-LABEL: @cas37(
// CHECK: cmpxchg ptr {{.*}} i64
_Bool cas37(_Atomic(S37) *p, S37 *e, S37 d) {
  return __c11_atomic_compare_exchange_strong(p, e, d, __ATOMIC_SEQ_CST,
                                              __ATOMIC_SEQ_CST);
}

// Bitwise RMW on a padded width keeps the direct atomicrmw: it is exact.
// CHECK-LABEL: @and37(
// CHECK: atomicrmw and ptr {{.*}} i64
// CHECK-NOT: cmpxchg
S37 and37(_Atomic(S37) *p, S37 v) {
  return __c11_atomic_fetch_and(p, v, __ATOMIC_SEQ_CST);
}

// Arithmetic RMW on a padded width becomes a compare-exchange loop, not a bare
// atomicrmw that would carry into the padding bits.
// CHECK-LABEL: @add37(
// CHECK: atomicrmw.start:
// CHECK: cmpxchg weak ptr {{.*}} i64
// CHECK-NOT: atomicrmw add
S37 add37(_Atomic(S37) *p, S37 v) {
  return __c11_atomic_fetch_add(p, v, __ATOMIC_SEQ_CST);
}

// Signed min is computed at the value width, so the sign bit is at bit N-1.
// CHECK-LABEL: @min37(
// CHECK: icmp sle i37
// CHECK: select i1
// CHECK: cmpxchg weak ptr {{.*}} i64
U37 min37(_Atomic(S37) *p, S37 v) {
  return __c11_atomic_fetch_min(p, v, __ATOMIC_SEQ_CST);
}

// No padding: direct atomicrmw, no loop.
// CHECK-LABEL: @add64(
// CHECK: atomicrmw add ptr {{.*}} i64
// CHECK-NOT: cmpxchg
S64 add64(_Atomic(S64) *p, S64 v) {
  return __c11_atomic_fetch_add(p, v, __ATOMIC_SEQ_CST);
}

// CHECK-LABEL: @add128(
// CHECK: atomicrmw add ptr {{.*}} i128
S128 add128(_Atomic(S128) *p, S128 v) {
  return __c11_atomic_fetch_add(p, v, __ATOMIC_SEQ_CST);
}

// Wide: no inline atomicrmw and no arbitrary-width __atomic_fetch_add libcall,
// so the loop calls __atomic_compare_exchange.
// CHECK-LABEL: @add256(
// CHECK: call {{.*}}@__atomic_compare_exchange
// CHECK-NOT: cmpxchg
S256 add256(_Atomic(S256) *p, S256 v) {
  return __c11_atomic_fetch_add(p, v, __ATOMIC_SEQ_CST);
}

// Wide bitwise also needs the loop: the wide path has no inline atomicrmw.
// CHECK-LABEL: @or256(
// CHECK: call {{.*}}@__atomic_compare_exchange
S256 or256(_Atomic(S256) *p, S256 v) {
  return __c11_atomic_fetch_or(p, v, __ATOMIC_SEQ_CST);
}
