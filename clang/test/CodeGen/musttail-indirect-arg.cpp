// RUN: %clang_cc1 -triple=riscv64-linux-gnu %s -emit-llvm -O1 -o - | FileCheck %s --check-prefix=COMMON
// RUN: %clang_cc1 -triple=aarch64-linux-gnu %s -emit-llvm -O1 -o - | FileCheck %s --check-prefix=COMMON
// RUN: %clang_cc1 -triple=loongarch64-linux-gnu %s -emit-llvm -O1 -o - | FileCheck %s --check-prefix=COMMON
// RUN: %clang_cc1 -triple=s390x-linux-gnu %s -emit-llvm -O1 -o - | FileCheck %s --check-prefix=COMMON
// RUN: %clang_cc1 -std=c++23 -triple=aarch64-linux-gnu %s -emit-llvm -O1 -o - | FileCheck %s --check-prefix=CXX23
// RUN: %clang_cc1 -triple=arm64e-apple-ios -fptrauth-calls -fptrauth-intrinsics %s -emit-llvm -O1 -o - | FileCheck %s --check-prefix=PTRAUTH

// C++ side of the musttail Indirect-arg fix. The call argument is typically
// a CXXConstructExpr invoking the trivial copy constructor; EmitCallArg
// hands the same-type glvalue source's LValue to EmitCall so the general
// path engages. Non-trivial copy or move constructors keep the existing
// agg.tmp path.

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
// COMMON: musttail call {{.*}} @_Z2C13Big({{.*}}, ptr {{[^,]*}} %a)

// P2: two distinct args.
struct Big C2(struct Big a, struct Big b);
struct Big P2(struct Big a, struct Big b) {
  [[clang::musttail]] return C2(a, b);
}
// COMMON-LABEL: define {{.*}} @_Z2P23BigS_(
// COMMON-NOT: llvm.memcpy
// COMMON: musttail call {{.*}} @_Z2C23BigS_({{.*}}, ptr {{[^,]*}} %a, ptr {{[^,]*}} %b)

// P3: swap. Pin the data flow (see musttail-indirect-arg.c): %a is captured
// before %b overwrites it, and the saved %a lands in %b.
struct Big C3(struct Big x, struct Big y);
struct Big P3(struct Big a, struct Big b) {
  [[clang::musttail]] return C3(b, a);
}
// COMMON-LABEL: define {{.*}} @_Z2P33BigS_(
// COMMON: [[SAVED:%musttail.copy[0-9.a-z]*]] = load {{.*}}, ptr %a,
// COMMON: @llvm.mem{{(cpy|move)}}{{.*}}(ptr {{[^,]*}} %a, ptr {{[^,]*}} %b,
// COMMON: store {{.*}} [[SAVED]], ptr %b,
// COMMON: musttail call {{.*}} @_Z2C33BigS_({{.*}}, ptr {{[^,]*}} %a, ptr {{[^,]*}} %b)

// P4: no trivial copy or move operation, so the argument is not relocated.
// Keep operator= declared: without it the implicit copy assignment stays
// trivial and EmitAggregateCopy would accept a bytewise copy.
struct NonTrivial {
  unsigned long long parts[4];
  NonTrivial(const NonTrivial &);
  NonTrivial &operator=(const NonTrivial &);
};
NonTrivial C4(NonTrivial a);
NonTrivial P4(NonTrivial a) {
  [[clang::musttail]] return C4(a);
}
// COMMON-LABEL: define {{.*}} @_Z2P410NonTrivial(
// COMMON: call {{.*}} @_ZN10NonTrivialC1ERKS_(ptr {{[^,]*}} [[TMP:%agg.tmp[0-9]*]], ptr {{[^,]*}} %a
// COMMON-NOT: musttail.copy
// COMMON: musttail call {{.*}} @_Z2C410NonTrivial({{.*}}, ptr {{[^,]*}} [[TMP]])

// P4b: only a copy ctor, so the implicit copy assignment is trivial. Still not
// relocatable: a copy ctor can record the object's own address.
struct CtorOnly {
  unsigned long long parts[4];
  CtorOnly(const CtorOnly &);
};
CtorOnly C4b(CtorOnly x, CtorOnly y);
CtorOnly P4b(CtorOnly a, CtorOnly b) {
  [[clang::musttail]] return C4b(b, a);
}
// COMMON-LABEL: define {{.*}} @_Z3P4b8CtorOnlyS_(
// COMMON-NOT: musttail.copy
// COMMON: musttail call {{.*}} @_Z3C4b8CtorOnlyS_(

// P4c: trivial_abi marks the type ABI-trivial, so relocation is allowed
// despite the non-trivial copy ctor.
struct __attribute__((trivial_abi)) TrivialAbi {
  unsigned long long parts[4];
  TrivialAbi(const TrivialAbi &);
};
TrivialAbi C4c(TrivialAbi x, TrivialAbi y);
TrivialAbi P4c(TrivialAbi a, TrivialAbi b) {
  [[clang::musttail]] return C4c(b, a);
}
// COMMON-LABEL: define {{.*}} @_Z3P4c10TrivialAbiS_(
// COMMON: [[SLOT0:%agg.tmp[0-9]*]] = alloca
// COMMON: [[SLOT1:%musttail.copy[0-9a-z.]*]] = alloca
// COMMON: @llvm.mem{{(cpy|move)}}{{.*}}(ptr {{[^,]*}} %a, ptr {{[^,]*}} [[SLOT0]], i64 32
// COMMON: @llvm.mem{{(cpy|move)}}{{.*}}(ptr {{[^,]*}} %b, ptr {{[^,]*}} [[SLOT1]], i64 32
// COMMON: musttail call {{.*}} @_Z3C4c10TrivialAbiS_({{.*}}, ptr {{[^,]*}} %a, ptr {{[^,]*}} %b)

// P4d: a virtual function makes the type non-trivially copyable. Pins the
// boundary so widening the relocation predicate has to update this.
struct Poly {
  unsigned long long parts[4];
  virtual void f();
};
Poly C4d(Poly x, Poly y);
Poly P4d(Poly a, Poly b) {
  [[clang::musttail]] return C4d(b, a);
}
// COMMON-LABEL: define {{.*}} @_Z3P4d4PolyS_(
// COMMON-NOT: musttail.copy
// COMMON: musttail call {{.*}} @_Z3C4d4PolyS_(

// P4e: a union is enough for EmitAggregateCopy, but the user copy operations
// still rule out a relocation.
union UnionCopy {
  unsigned long long parts[4];
  UnionCopy(const UnionCopy &);
  UnionCopy &operator=(const UnionCopy &);
};
UnionCopy C4e(UnionCopy x, UnionCopy y);
UnionCopy P4e(UnionCopy a, UnionCopy b) {
  [[clang::musttail]] return C4e(b, a);
}
// COMMON-LABEL: define {{.*}} @_Z3P4e9UnionCopyS_(
// COMMON-NOT: musttail.copy
// COMMON: musttail call {{.*}} @_Z3C4e9UnionCopyS_(

#ifdef __PTRAUTH__
// P4f: an address-discriminated __ptrauth member is signed with the object's
// own address, so relocating the bytes would leave the signature bound to the
// old one.
struct Signed {
  int *__ptrauth(2, 1, 42) p;
  unsigned long long a, b, c;
};
Signed C4f(Signed x, Signed y);
Signed P4f(Signed a, Signed b) {
  [[clang::musttail]] return C4f(b, a);
}
// PTRAUTH-LABEL: define {{.*}} @_Z3P4f6SignedS_(
// PTRAUTH-NOT: musttail.copy
// PTRAUTH: musttail call {{.*}} @_Z3C4f6SignedS_(
#endif

// P5: modify-then-forward.
struct Big C5(struct Big a);
struct Big P5(struct Big a) {
  a.a += 1;
  [[clang::musttail]] return C5(a);
}
// COMMON-LABEL: define {{.*}} @_Z2P53Big(
// COMMON: musttail call {{.*}} @_Z2C53Big({{.*}}, ptr {{[^,]*}} %a)

// P6: musttail behind a branch.
struct Big C6(struct Big a, int cond);
struct Big P6(struct Big a, int cond) {
  if (cond)
    [[clang::musttail]] return C6(a, cond);
  return a;
}
// COMMON-LABEL: define {{.*}} @_Z2P63Bigi(
// COMMON: musttail call {{.*}} @_Z2C63Bigi({{.*}}, ptr {{[^,]*}} %a,

// P7: same arg to two slots. Slot 0 forwards %a; slot 1 memcpys *%a into the
// i=1 incoming pointer %b and forwards %b.
struct Big C7(struct Big x, struct Big y);
struct Big P7(struct Big a, struct Big b) {
  [[clang::musttail]] return C7(a, a);
}
// COMMON-LABEL: define {{.*}} @_Z2P73BigS_(
// COMMON: llvm.mem{{(cpy|move)}}{{.*}}(ptr {{[^,]*}} %b, ptr {{[^,]*}} %a,
// COMMON: musttail call {{.*}} @_Z2C73BigS_({{.*}}, ptr {{[^,]*}} %a, ptr {{[^,]*}} %b)

// P8: local source. Copied into the incoming %a, then %a forwarded.
struct Big C8(struct Big a);
struct Big P8(struct Big a) {
  struct Big local = {1, 2, 3, 4};
  [[clang::musttail]] return C8(local);
}
// COMMON-LABEL: define {{.*}} @_Z2P83Big(
// COMMON: llvm.mem{{(cpy|move)}}{{.*}}(ptr {{[^,]*}} %a, ptr {{.*}}
// COMMON: musttail call {{.*}} @_Z2C83Big({{.*}}, ptr {{[^,]*}} %a)

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
// COMMON: musttail call {{.*}} @_Z3C10i3BigiS_({{.*}}, i32 {{.*}} %x1, ptr {{[^,]*}} %s1, i32 {{.*}} %x2, ptr {{[^,]*}} %s2)

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
// COMMON: musttail call {{.*}} @_ZN1S1fE3Big({{.*}}, ptr {{.*}}, ptr {{[^,]*}} %a)

// P13: mixed source kinds (local + incoming parameter).
struct Big C13(struct Big x, struct Big y);
struct Big P13(struct Big a, struct Big b) {
  struct Big local = {1, 2, 3, 4};
  [[clang::musttail]] return C13(local, a);
}
// COMMON-LABEL: define {{.*}} @_Z3P133BigS_(
// COMMON-NOT: byval-temp
// COMMON: %musttail.copy{{[0-9.a-z]*}} =
// COMMON: musttail call {{.*}} @_Z3C133BigS_({{.*}}, ptr {{[^,]*}} %a, ptr {{[^,]*}} %b)

// P17: same arg to three slots (generalization of P7). Both copied slots
// take their value from %a: %b via the memmove, %c via the captured load.
struct Big C17(struct Big x, struct Big y, struct Big z);
struct Big P17(struct Big a, struct Big b, struct Big c) {
  [[clang::musttail]] return C17(a, a, a);
}
// COMMON-LABEL: define {{.*}} @_Z3P173BigS_S_(
// COMMON: [[SAVED:%musttail.copy[0-9.a-z]*]] = load {{.*}}, ptr %a,
// COMMON: @llvm.mem{{(cpy|move)}}{{.*}}(ptr {{[^,]*}} %b, ptr {{[^,]*}} %a,
// COMMON: store {{.*}} [[SAVED]], ptr %c,
// COMMON: musttail call {{.*}} @_Z3C173BigS_S_({{.*}}, ptr {{[^,]*}} %a, ptr {{[^,]*}} %b, ptr {{[^,]*}} %c)

// P18: member of a global as the source. Forwarded with no agg.tmp; the copy
// lands directly in the incoming %a.
struct Wrap {
  struct Big inner;
};
extern Wrap gw;
struct Big C18(struct Big a);
struct Big P18(struct Big a) {
  [[clang::musttail]] return C18(gw.inner);
}
// COMMON-LABEL: define {{.*}} @_Z3P183Big(
// COMMON-NOT: %agg.tmp
// COMMON: @llvm.mem{{(cpy|move)}}{{.*}}(ptr {{[^,]*}} %a, ptr {{[^,]*}} @gw, i64 32
// COMMON: musttail call {{.*}} @_Z3C183Big({{.*}}, ptr {{[^,]*}} %a)

// P19: deref of a global pointer. The address computation reads mutable
// state, so the bytes are captured at argument position (no forwarding);
// the temp still routes through the incoming parameter.
extern struct Big *gp;
struct Big C19(struct Big a);
struct Big P19(struct Big a) {
  [[clang::musttail]] return C19(*gp);
}
// COMMON-LABEL: define {{.*}} @_Z3P193Big(
// COMMON-NOT: %agg.tmp
// COMMON: [[SRC:%[0-9a-z.]+]] = load ptr, ptr @gp
// COMMON: @llvm.mem{{(cpy|move)}}{{.*}}(ptr {{[^,]*}} %a, ptr {{[^,]*}} [[SRC]], i64 32
// COMMON: musttail call {{.*}} @_Z3C193Big({{.*}}, ptr {{[^,]*}} %a)

// P20: derived-to-base source. The base subobject sits at offset 8 in Der;
// the forwarded address must carry that adjustment.
struct Pad {
  unsigned long long p;
};
struct Der : Pad, Big {};
extern Der gd;
struct Big C20(struct Big a);
struct Big P20(struct Big a) {
  [[clang::musttail]] return C20(gd);
}
// COMMON-LABEL: define {{.*}} @_Z3P203Big(
// COMMON-NOT: %agg.tmp
// COMMON: @llvm.mem{{(cpy|move)}}{{.*}}(ptr {{[^,]*}} %a, ptr {{[^,]*}} getelementptr inbounds {{(nuw )?}}(i8, ptr @gd, i64 8), i64 32
// COMMON: musttail call {{.*}} @_Z3C203Big({{.*}}, ptr {{[^,]*}} %a)

// P21: impure source with a side-effecting second argument. The source bytes
// are read at argument position, before bump() runs, so the argument's
// evaluation is not interleaved with the other argument's ([expr.call]/8).
extern int bump();
struct Big C21(struct Big x, int y);
struct Big P21(struct Big a, int b) {
  [[clang::musttail]] return C21(*gp, bump());
}
// COMMON-LABEL: define {{.*}} @_Z3P213Bigi(
// COMMON: [[SRC:%[0-9a-z.]+]] = load ptr, ptr @gp
// COMMON-NOT: @_Z4bumpv
// COMMON: [[VAL:%[0-9a-z.]+]] = load {{.*}}, ptr [[SRC]]
// COMMON: call {{.*}} @_Z4bumpv()
// COMMON: store {{.*}} [[VAL]], ptr %a
// COMMON: musttail call {{.*}} @_Z3C213Bigi({{.*}}, ptr {{[^,]*}} %a,

// P22: an overloaded operator keeps the built-in operand order
// ([over.match.oper]/2), so forwarding must not defer the left operand's read
// past the right operand's side effect.
struct Big operator<<(struct Big x, struct Big y);
struct Big P22(struct Big a, struct Big b) {
  [[clang::musttail]] return a << Big{a.a = 10};
}
// COMMON-LABEL: define {{.*}} @_Z3P223BigS_(
// COMMON: [[SNAP:%[0-9a-z.]+]] = load <4 x i64>, ptr %a
// COMMON: store i64 10, ptr %a
// COMMON: store <4 x i64> [[SNAP]], ptr %a
// COMMON: musttail call {{.*}} @_Zls3BigS_({{.*}}, ptr {{[^,]*}} %a, ptr {{[^,]*}} %b)

// P24: [expr.assign]/1 sequences an assignment's right operand first, so the
// deferred read of the right operand is the one that must not move.
struct Big operator+=(struct Big x, struct Big y);
struct Big mutate(struct Big *p);
struct Big P24(struct Big a, struct Big b) {
  [[clang::musttail]] return mutate(&b) += b;
}
// COMMON-LABEL: define {{.*}} @_Z3P243BigS_(
// COMMON: [[SNAP:%[0-9a-z.]+]] = load <4 x i64>, ptr %b
// COMMON: call {{.*}} @_Z6mutateP3Big(
// COMMON: store <4 x i64> [[SNAP]], ptr %b
// COMMON: musttail call {{.*}} @_ZpL3BigS_({{.*}}, ptr {{[^,]*}} %a, ptr {{[^,]*}} %b)

// P25: the other operand mutates the source through an opaque callee, so
// nothing in the expression names it. Whether the source is reachable from the
// other operand cannot decide this.
struct Big bumpg();
struct Big P25(struct Big a, struct Big b) {
  [[clang::musttail]] return a << bumpg();
}
// COMMON-LABEL: define {{.*}} @_Z3P253BigS_(
// COMMON: [[SNAP:%[0-9a-z.]+]] = load <4 x i64>, ptr %a
// COMMON: call {{.*}} @_Z5bumpgv(
// COMMON: store <4 x i64> [[SNAP]], ptr %a
// COMMON: musttail call {{.*}} @_Zls3BigS_({{.*}}, ptr {{[^,]*}} %a, ptr {{[^,]*}} %b)

// P26: the right-operand-first rule of [expr.assign]/1 through a member
// operator+=, so the read of %rhs must precede bumpacc().
struct Acc {
  unsigned long long v;
  Acc &operator+=(struct Big rhs);
  Acc &tailadd(struct Big rhs);
};
Acc &bumpacc(struct Big *p);
Acc &Acc::tailadd(struct Big rhs) {
  [[clang::musttail]] return bumpacc(&rhs) += rhs;
}
// COMMON-LABEL: define {{.*}} @_ZN3Acc7tailaddE3Big(
// COMMON: [[SNAP:%[0-9a-z.]+]] = load <4 x i64>, ptr %rhs
// COMMON: call {{.*}} @_Z7bumpaccP3Big(
// COMMON: store <4 x i64> [[SNAP]], ptr %rhs
// COMMON: musttail call {{.*}} @_ZN3AccpLE3Big({{.*}}, ptr {{[^,]*}} %rhs)

// P27: the prescribed-order call is an argument inside the musttail return,
// not the tail call itself.
struct Big bumpbig();
int use27(struct Big v);
int C27(int x);
int P27(int x) {
  [[clang::musttail]] return C27(use27(gw.inner << bumpbig()));
}
// COMMON-LABEL: define {{.*}} @_Z3P27i(
// COMMON: @llvm.mem{{(cpy|move)}}{{.*}}(ptr {{[^,]*}} [[LHS:%agg.tmp[0-9]*]], ptr {{[^,]*}} @gw, i64 32
// COMMON: call {{.*}} @_Z7bumpbigv(
// COMMON: call {{.*}} @_Zls3BigS_({{.*}}, ptr {{[^,]*}} [[LHS]],
// COMMON: musttail call {{.*}} @_Z3C27i(

// P28: prescribed order with the source in the second incoming slot, so both
// argument slots are relocated and the read still precedes the mutation.
struct Big operator>>(struct Big x, struct Big y);
struct Big P28(struct Big a, struct Big b) {
  [[clang::musttail]] return b >> Big{b.a = 10};
}
// COMMON-LABEL: define {{.*}} @_Z3P283BigS_(
// COMMON: [[SNAP:%[0-9a-z.]+]] = load <4 x i64>, ptr %b
// COMMON: store i64 10, ptr %b
// COMMON: store <4 x i64> [[SNAP]], ptr %a
// COMMON: musttail call {{.*}} @_Zrs3BigS_({{.*}}, ptr {{[^,]*}} %a, ptr {{[^,]*}} %b)

#if __cplusplus >= 202302L
// P23: the same rule for an operator with no explicit EvaluationOrder case.
// A subscript operator's object parameter is sequenced before the index
// ([expr.sub]/1).
struct Sub {
  unsigned long long a, b, c, d;
  Sub operator[](this Sub self, int i);
  Sub tail(this Sub self, int i);
};
int bump(Sub *p);
Sub Sub::tail(this Sub self, int i) {
  [[clang::musttail]] return self[bump(&self)];
}
// CXX23-LABEL: define {{.*}} @_ZNH3Sub4tailES_i(
// CXX23: [[SNAP:%[0-9a-z.]+]] = load <4 x i64>, ptr %self
// CXX23: call {{.*}} @_Z4bumpP3Sub(
// CXX23: store <4 x i64> [[SNAP]], ptr %self
// CXX23: musttail call {{.*}} @_ZNH3SubixES_i({{.*}}, ptr {{[^,]*}} %self, i32 {{.*}})
#endif
