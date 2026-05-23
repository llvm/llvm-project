// Test that Clang forwards incoming Indirect parameters across musttail calls
// for C++ struct-by-value arguments with trivially-copyable types. Companion to
// musttail-indirect-arg.c (the C side of the same fix) and musttail-sret.cpp
// (the SRet precedent in a96c14eeb8fc).
//
// C++ goes through a different EmitCallArg path than C: the call argument is a
// CXXConstructExpr invoking the implicit copy constructor, which would
// otherwise materialize an agg.tmp before EmitCall. For musttail with a
// trivially-copyable parameter forwarded directly, the copy is elided so the
// helper in EmitCall can forward the incoming llvm::Argument.

// RUN: %clang_cc1 -triple=riscv64-linux-gnu %s -emit-llvm -O1 -o - | FileCheck %s --check-prefix=COMMON
// RUN: %clang_cc1 -triple=aarch64-linux-gnu %s -emit-llvm -O1 -o - | FileCheck %s --check-prefix=COMMON
// RUN: %clang_cc1 -triple=loongarch64-linux-gnu %s -emit-llvm -O1 -o - | FileCheck %s --check-prefix=COMMON
// RUN: %clang_cc1 -triple=s390x-linux-gnu %s -emit-llvm -O1 -o - | FileCheck %s --check-prefix=COMMON

// A trivially-copyable struct large enough to land on the indirect-arg path
// on RV64, AArch64, LoongArch64, SystemZ.
struct Big {
  unsigned long long a, b, c, d;
};

// Plain forward: caller(B) musttails callee(B). No agg.tmp copy, no
// byval-temp; the incoming parameter %a is forwarded directly.
struct Big C1(struct Big a);
struct Big P1(struct Big a) {
  [[clang::musttail]] return C1(a);
}
// COMMON-LABEL: define {{.*}} @_Z2P13Big(
// COMMON-NOT: = alloca {{.*}}struct.Big
// COMMON: musttail call {{.*}} @_Z2C13Big({{.*}} %a)

// Two args, same forwarding.
struct Big C2(struct Big a, struct Big b);
struct Big P2(struct Big a, struct Big b) {
  [[clang::musttail]] return C2(a, b);
}
// COMMON-LABEL: define {{.*}} @_Z2P23BigS_(
// COMMON-NOT: = alloca {{.*}}struct.Big
// COMMON: musttail call {{.*}} @_Z2C23BigS_({{.*}} %a, {{.*}} %b)

// Swapped args.
struct Big C3(struct Big x, struct Big y);
struct Big P3(struct Big a, struct Big b) {
  [[clang::musttail]] return C3(b, a);
}
// COMMON-LABEL: define {{.*}} @_Z2P33BigS_(
// COMMON-NOT: = alloca {{.*}}struct.Big
// COMMON: musttail call {{.*}} @_Z2C33BigS_({{.*}} %b, {{.*}} %a)

// Non-trivial copy constructor: the trivial-copy elision must NOT engage.
// Existing path materializes the agg.tmp (the user-defined copy ctor has
// observable behavior).
struct NonTrivial {
  unsigned long long parts[4];
  NonTrivial(const NonTrivial &);
};
NonTrivial C4(NonTrivial a);
NonTrivial P4(NonTrivial a) {
  [[clang::musttail]] return C4(a);
}
// COMMON-LABEL: define {{.*}} @_Z2P410NonTrivial(
// The user-defined copy ctor IS called (the agg.tmp pattern still happens):
// COMMON: call {{.*}} @_ZN10NonTrivialC1ERKS_

// Non-musttail tail call: trivial-copy elision must NOT engage; the regular
// agg.tmp copy is still emitted.
struct Big C5(struct Big a);
struct Big P5(struct Big a) {
  return C5(a);
}
// COMMON-LABEL: define {{.*}} @_Z2P53Big(
// COMMON-NOT: musttail
