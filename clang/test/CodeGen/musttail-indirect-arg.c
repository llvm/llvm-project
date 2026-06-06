// RUN: %clang_cc1 -triple=riscv64-linux-gnu %s -emit-llvm -O1 -o - | FileCheck %s --check-prefix=COMMON
// RUN: %clang_cc1 -triple=aarch64-linux-gnu %s -emit-llvm -O1 -o - | FileCheck %s --check-prefix=COMMON
// RUN: %clang_cc1 -triple=loongarch64-linux-gnu %s -emit-llvm -O1 -o - | FileCheck %s --check-prefix=COMMON
// RUN: %clang_cc1 -triple=s390x-linux-gnu %s -emit-llvm -O1 -o - | FileCheck %s --check-prefix=COMMON

// Musttail calls with struct-by-value args must route each value through the
// matching incoming Indirect parameter's storage, not a local alloca that
// dangles past the tail-call frame teardown. For each Indirect slot i: if
// the source IS the i-th incoming pointer, pass it; otherwise memcpy the
// source into the i-th incoming pointer and pass that. Each call slot ends
// up with a distinct pointer in the caller's caller's frame.

// Plain Indirect-ABI struct on the targets above.
struct Big {
  unsigned long long a, b, c, d;
};

// P1: simple forward.
struct Big C1(struct Big a);
struct Big P1(struct Big a) {
  __attribute__((musttail)) return C1(a);
}
// COMMON-LABEL: define {{.*}} @P1(
// COMMON-NOT: = alloca {{.*}}struct.Big
// COMMON: musttail call {{.*}} @C1({{.*}}, ptr {{.*}} %a)

// P2: two distinct incoming sources.
struct Big C2(struct Big a, struct Big b);
struct Big P2(struct Big a, struct Big b) {
  __attribute__((musttail)) return C2(a, b);
}
// COMMON-LABEL: define {{.*}} @P2(
// COMMON-NOT: = alloca {{.*}}struct.Big
// COMMON-NOT: llvm.memcpy
// COMMON: musttail call {{.*}} @C2({{.*}}, ptr {{.*}} %a, ptr {{.*}} %b)

// P3: swap. Slot 0's dst is %a (positional), so the value of %b is copied
// into %a; symmetrically for slot 1. The scratch alloca catches the in-
// place-write regression: v1 would emit memcpy(%a, %b); memcpy(%b, %a) and
// both slots would end up with orig_b. v2 captures at least one source
// into a scratch first; we assert the scratch is present.
struct Big C3(struct Big x, struct Big y);
struct Big P3(struct Big a, struct Big b) {
  __attribute__((musttail)) return C3(b, a);
}
// COMMON-LABEL: define {{.*}} @P3(
// COMMON: %musttail.copy{{[0-9.a-z]*}} =
// COMMON: musttail call {{.*}} @C3({{.*}}, ptr {{.*}} %a, ptr {{.*}} %b)

// P5: caller mutates the parameter before the musttail. The mutation lands
// at the incoming pointer the callee receives.
struct Big C5(struct Big a);
struct Big P5(struct Big a) {
  a.a += 1;
  __attribute__((musttail)) return C5(a);
}
// COMMON-LABEL: define {{.*}} @P5(
// COMMON: musttail call {{.*}} @C5({{.*}}, ptr {{.*}} %a)

// P6: musttail in a non-entry block.
struct Big C6(struct Big a, int cond);
struct Big P6(struct Big a, int cond) {
  if (cond)
    __attribute__((musttail)) return C6(a, cond);
  return a;
}
// COMMON-LABEL: define {{.*}} @P6(
// COMMON: musttail call {{.*}} @C6({{.*}}, ptr {{.*}} %a,

// P7: same arg to two slots. C ABI requires distinct storage per by-value
// param, so slot 1 cannot share %a's pointer. Slot 0 forwards %a; slot 1
// memcpys *%a into the i=1 incoming pointer %b and forwards %b.
struct Big C7(struct Big x, struct Big y);
struct Big P7(struct Big a, struct Big b) {
  __attribute__((musttail)) return C7(a, a);
}
// COMMON-LABEL: define {{.*}} @P7(
// COMMON: llvm.mem{{(cpy|move)}}{{.*}}(ptr {{.*}} %b, ptr {{.*}} %a,
// COMMON: musttail call {{.*}} @C7({{.*}}, ptr {{.*}} %a, ptr {{.*}} %b)

// P8: local source. The local lives in our frame; copy it into %a, forward %a.
struct Big C8(struct Big a);
struct Big P8(struct Big a) {
  struct Big local = {1, 2, 3, 4};
  __attribute__((musttail)) return C8(local);
}
// COMMON-LABEL: define {{.*}} @P8(
// COMMON: llvm.mem{{(cpy|move)}}{{.*}}(ptr {{.*}} %a, ptr {{.*}}
// COMMON: musttail call {{.*}} @C8({{.*}}, ptr {{.*}} %a)

// P9: non-musttail tail call (existing path).
struct Big C9(struct Big a);
struct Big P9(struct Big a) {
  return C9(a);
}
// COMMON-LABEL: define {{.*}} @P9(
// COMMON-NOT: musttail

// P10: mixed direct + indirect.
struct Big C10(int x1, struct Big s1, int x2, struct Big s2);
struct Big P10(int x1, struct Big s1, int x2, struct Big s2) {
  __attribute__((musttail)) return C10(x1, s1, x2, s2);
}
// COMMON-LABEL: define {{.*}} @P10(
// COMMON-NOT: = alloca {{.*}}struct.Big
// COMMON: musttail call {{.*}} @C10({{.*}}, i32 {{.*}} %x1, ptr {{.*}} %s1, i32 {{.*}} %x2, ptr {{.*}} %s2)

// P11: many args, including stack-spilled ones on the target ABIs above.
struct Big C11(struct Big s1, struct Big s2, struct Big s3, struct Big s4,
               struct Big s5, struct Big s6, struct Big s7, struct Big s8,
               struct Big s9, struct Big s10);
struct Big P11(struct Big a1, struct Big a2, struct Big a3, struct Big a4,
               struct Big a5, struct Big a6, struct Big a7, struct Big a8,
               struct Big a9, struct Big a10) {
  __attribute__((musttail)) return C11(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10);
}
// COMMON-LABEL: define {{.*}} @P11(
// COMMON-NOT: = alloca {{.*}}struct.Big
// COMMON: musttail call {{.*}} @C11(
// COMMON-SAME: ptr {{.*}} %a1, ptr {{.*}} %a2, ptr {{.*}} %a3, ptr {{.*}} %a4
// COMMON-SAME: ptr {{.*}} %a5, ptr {{.*}} %a6, ptr {{.*}} %a7, ptr {{.*}} %a8
// COMMON-SAME: ptr {{.*}} %a9, ptr {{.*}} %a10

// P12: over-aligned struct.
struct __attribute__((aligned(32))) AlignedBig {
  unsigned long long a, b, c, d;
};
struct AlignedBig C12(struct AlignedBig a);
struct AlignedBig P12(struct AlignedBig a) {
  __attribute__((musttail)) return C12(a);
}
// COMMON-LABEL: define {{.*}} @P12(
// COMMON: musttail call {{.*}} @C12({{.*}}, ptr {{.*}} %a)

// P13: mixed source kinds within Indirect slots. Slot 0's source is a local;
// slot 1's source is an incoming parameter. Both routes engage in the same
// call: local-source case (would have dangled under v1's byval-temp path)
// and forward case must coexist with two-phase ordering.
struct Big C13(struct Big x, struct Big y);
struct Big P13(struct Big a, struct Big b) {
  struct Big local = {1, 2, 3, 4};
  __attribute__((musttail)) return C13(local, a);
}
// COMMON-LABEL: define {{.*}} @P13(
// COMMON-NOT: byval-temp
// COMMON: %musttail.copy{{[0-9.a-z]*}} =
// COMMON: musttail call {{.*}} @C13({{.*}}, ptr {{.*}} %a, ptr {{.*}} %b)

// P17: same arg to three slots (generalization of P7).
struct Big C17(struct Big x, struct Big y, struct Big z);
struct Big P17(struct Big a, struct Big b, struct Big c) {
  __attribute__((musttail)) return C17(a, a, a);
}
// COMMON-LABEL: define {{.*}} @P17(
// The scratch presence catches the in-place-write regression: a one-pass
// emit would memcpy(%b, %a); memcpy(%c, %a) directly with no scratch.
// COMMON: %musttail.copy{{[0-9.a-z]*}} =
// COMMON: musttail call {{.*}} @C17({{.*}}, ptr {{.*}} %a, ptr {{.*}} %b, ptr {{.*}} %c)
