// RUN: %clang_cc1 -triple=riscv64-linux-gnu %s -emit-llvm -O1 -o - | FileCheck %s --check-prefix=COMMON
// RUN: %clang_cc1 -triple=aarch64-linux-gnu %s -emit-llvm -O1 -o - | FileCheck %s --check-prefix=COMMON
// RUN: %clang_cc1 -triple=loongarch64-linux-gnu %s -emit-llvm -O1 -o - | FileCheck %s --check-prefix=COMMON
// RUN: %clang_cc1 -triple=s390x-linux-gnu %s -emit-llvm -O1 -o - | FileCheck %s --check-prefix=COMMON

// C++ side of the musttail Indirect-arg fix. The call argument is typically
// a CXXConstructExpr invoking the trivial copy constructor; EmitCallArg
// detects the trivial-copy-from-DeclRefExpr case under musttail and hands
// the source LValue to EmitCall so the general path engages. Non-trivial
// copy or move constructors keep the existing agg.tmp path.

struct Big {
  unsigned long long a, b, c, d;
};

// P1: simple forward.
struct Big C1(struct Big a);
struct Big P1(struct Big a) {
  [[clang::musttail]] return C1(a);
}
// COMMON-LABEL: define {{.*}} @_Z2P13Big(
// COMMON-NOT: = alloca {{.*}}struct.Big
// COMMON: musttail call {{.*}} @_Z2C13Big({{.*}}, ptr {{.*}} %a)

// P2: two distinct args.
struct Big C2(struct Big a, struct Big b);
struct Big P2(struct Big a, struct Big b) {
  [[clang::musttail]] return C2(a, b);
}
// COMMON-LABEL: define {{.*}} @_Z2P23BigS_(
// COMMON-NOT: llvm.memcpy
// COMMON: musttail call {{.*}} @_Z2C23BigS_({{.*}}, ptr {{.*}} %a, ptr {{.*}} %b)

// P3: swap. Asserts the scratch alloca to catch the in-place-write
// regression (see musttail-indirect-arg.c).
struct Big C3(struct Big x, struct Big y);
struct Big P3(struct Big a, struct Big b) {
  [[clang::musttail]] return C3(b, a);
}
// COMMON-LABEL: define {{.*}} @_Z2P33BigS_(
// COMMON: %musttail.copy{{[0-9.a-z]*}} =
// COMMON: musttail call {{.*}} @_Z2C33BigS_({{.*}}, ptr {{.*}} %a, ptr {{.*}} %b)

// P4: non-trivial copy constructor. The trivial-copy gate must NOT engage;
// the user-defined copy ctor IS called. Dangling-stack bug in this corner
// remains (out of scope).
struct NonTrivial {
  unsigned long long parts[4];
  NonTrivial(const NonTrivial &);
};
NonTrivial C4(NonTrivial a);
NonTrivial P4(NonTrivial a) {
  [[clang::musttail]] return C4(a);
}
// COMMON-LABEL: define {{.*}} @_Z2P410NonTrivial(
// COMMON: call {{.*}} @_ZN10NonTrivialC1ERKS_

// P5: modify-then-forward.
struct Big C5(struct Big a);
struct Big P5(struct Big a) {
  a.a += 1;
  [[clang::musttail]] return C5(a);
}
// COMMON-LABEL: define {{.*}} @_Z2P53Big(
// COMMON: musttail call {{.*}} @_Z2C53Big({{.*}}, ptr {{.*}} %a)

// P6: musttail behind a branch.
struct Big C6(struct Big a, int cond);
struct Big P6(struct Big a, int cond) {
  if (cond)
    [[clang::musttail]] return C6(a, cond);
  return a;
}
// COMMON-LABEL: define {{.*}} @_Z2P63Bigi(
// COMMON: musttail call {{.*}} @_Z2C63Bigi({{.*}}, ptr {{.*}} %a,

// P7: same arg to two slots. Slot 0 forwards %a; slot 1 memcpys *%a into the
// i=1 incoming pointer %b and forwards %b.
struct Big C7(struct Big x, struct Big y);
struct Big P7(struct Big a, struct Big b) {
  [[clang::musttail]] return C7(a, a);
}
// COMMON-LABEL: define {{.*}} @_Z2P73BigS_(
// COMMON: llvm.mem{{(cpy|move)}}{{.*}}(ptr {{.*}} %b, ptr {{.*}} %a,
// COMMON: musttail call {{.*}} @_Z2C73BigS_({{.*}}, ptr {{.*}} %a, ptr {{.*}} %b)

// P8: local source. Copied into the incoming %a, then %a forwarded.
struct Big C8(struct Big a);
struct Big P8(struct Big a) {
  struct Big local = {1, 2, 3, 4};
  [[clang::musttail]] return C8(local);
}
// COMMON-LABEL: define {{.*}} @_Z2P83Big(
// COMMON: llvm.mem{{(cpy|move)}}{{.*}}(ptr {{.*}} %a, ptr {{.*}}
// COMMON: musttail call {{.*}} @_Z2C83Big({{.*}}, ptr {{.*}} %a)

// P9: non-musttail tail call (existing path).
struct Big C9(struct Big a);
struct Big P9(struct Big a) {
  return C9(a);
}
// COMMON-LABEL: define {{.*}} @_Z2P93Big(
// COMMON-NOT: musttail

// P10: mixed direct + indirect.
struct Big C10(int x1, struct Big s1, int x2, struct Big s2);
struct Big P10(int x1, struct Big s1, int x2, struct Big s2) {
  [[clang::musttail]] return C10(x1, s1, x2, s2);
}
// COMMON-LABEL: define {{.*}} @_Z3P10i3BigiS_(
// COMMON-NOT: = alloca {{.*}}struct.Big
// COMMON: musttail call {{.*}} @_Z3C10i3BigiS_({{.*}}, i32 {{.*}} %x1, ptr {{.*}} %s1, i32 {{.*}} %x2, ptr {{.*}} %s2)

// P11: many args (stack spill on the target ABIs above).
struct Big C11(struct Big s1, struct Big s2, struct Big s3, struct Big s4,
               struct Big s5, struct Big s6, struct Big s7, struct Big s8,
               struct Big s9, struct Big s10);
struct Big P11(struct Big a1, struct Big a2, struct Big a3, struct Big a4,
               struct Big a5, struct Big a6, struct Big a7, struct Big a8,
               struct Big a9, struct Big a10) {
  [[clang::musttail]] return C11(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10);
}
// COMMON-LABEL: define {{.*}} @_Z3P113BigS_S_S_S_S_S_S_S_S_(
// COMMON-NOT: = alloca {{.*}}struct.Big
// COMMON: musttail call {{.*}} @_Z3C113BigS_S_S_S_S_S_S_S_S_(

// P16: member function. (P15 lambda case skipped: Sema currently rejects
// musttail from a lambda's operator() to a non-member function, #119152.)
struct S {
  struct Big f(struct Big a);
  struct Big P16(struct Big a);
};
struct Big S::P16(struct Big a) {
  [[clang::musttail]] return f(a);
}
// COMMON-LABEL: define {{.*}} @_ZN1S3P16E3Big(
// COMMON-NOT: = alloca {{.*}}struct.Big
// COMMON: musttail call {{.*}} @_ZN1S1fE3Big({{.*}}, ptr {{.*}}, ptr {{.*}} %a)

// P13: mixed source kinds (local + incoming parameter).
struct Big C13(struct Big x, struct Big y);
struct Big P13(struct Big a, struct Big b) {
  struct Big local = {1, 2, 3, 4};
  [[clang::musttail]] return C13(local, a);
}
// COMMON-LABEL: define {{.*}} @_Z3P133BigS_(
// COMMON-NOT: byval-temp
// COMMON: %musttail.copy{{[0-9.a-z]*}} =
// COMMON: musttail call {{.*}} @_Z3C133BigS_({{.*}}, ptr {{.*}} %a, ptr {{.*}} %b)

// P17: same arg to three slots (generalization of P7).
struct Big C17(struct Big x, struct Big y, struct Big z);
struct Big P17(struct Big a, struct Big b, struct Big c) {
  [[clang::musttail]] return C17(a, a, a);
}
// COMMON-LABEL: define {{.*}} @_Z3P173BigS_S_(
// COMMON: %musttail.copy{{[0-9.a-z]*}} =
// COMMON: musttail call {{.*}} @_Z3C173BigS_S_({{.*}}, ptr {{.*}} %a, ptr {{.*}} %b, ptr {{.*}} %c)
