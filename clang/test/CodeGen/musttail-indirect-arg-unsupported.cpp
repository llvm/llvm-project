// RUN: %clang_cc1 -triple=riscv64-linux-gnu -verify -emit-llvm-only %s
// RUN: %clang_cc1 -triple=aarch64-linux-gnu -verify -emit-llvm-only %s
// RUN: %clang_cc1 -triple=loongarch64-linux-gnu -verify -emit-llvm-only %s
// RUN: %clang_cc1 -triple=s390x-linux-gnu -verify -emit-llvm-only %s
// RUN: %clang_cc1 -triple=arm64e-apple-ios -fptrauth-calls -fptrauth-intrinsics -verify -emit-llvm-only %s
// RUN: %clang_cc1 -triple=riscv64-linux-gnu -DCASE_NONTRIVIAL_COPY_CTOR -verify -emit-llvm-only %s
// RUN: %clang_cc1 -triple=riscv64-linux-gnu -DCASE_POLYMORPHIC -verify -emit-llvm-only %s
// RUN: %clang_cc1 -triple=riscv64-linux-gnu -DCASE_NONTRIVIAL_UNION -verify -emit-llvm-only %s
// RUN: %clang_cc1 -triple=arm64e-apple-ios -fptrauth-calls -fptrauth-intrinsics -DCASE_PTRAUTH -verify -emit-llvm-only %s

#if defined(CASE_NONTRIVIAL_COPY_CTOR)
struct CtorOnly {
  unsigned long long parts[4];
  CtorOnly(const CtorOnly &);
};
struct Trivial {
  unsigned long long parts[4];
};
CtorOnly C4b(Trivial x, CtorOnly y);
CtorOnly P4b(Trivial a, CtorOnly b) {
  // expected-error@+1 {{'musttail' call cannot safely forward this indirect argument}}
  [[clang::musttail]] return C4b(a, b);
}
#elif defined(CASE_POLYMORPHIC)
struct Poly {
  unsigned long long parts[4];
  virtual void f();
};
Poly C4d(Poly x);
Poly P4d(Poly a) {
  // expected-error@+1 {{'musttail' call cannot safely forward this indirect argument}}
  [[clang::musttail]] return C4d(a);
}
#elif defined(CASE_NONTRIVIAL_UNION)
union UnionCopy {
  unsigned long long parts[4];
  UnionCopy(const UnionCopy &);
  UnionCopy &operator=(const UnionCopy &);
};
UnionCopy C4e(UnionCopy x);
UnionCopy P4e(UnionCopy a) {
  // expected-error@+1 {{'musttail' call cannot safely forward this indirect argument}}
  [[clang::musttail]] return C4e(a);
}
#elif defined(CASE_PTRAUTH)
struct Signed {
  int *__ptrauth(2, 1, 42) p;
  unsigned long long a, b, c;
};
Signed C4f(Signed x);
Signed P4f(Signed a) {
  // expected-error@+1 {{'musttail' call cannot safely forward this indirect argument}}
  [[clang::musttail]] return C4f(a);
}
#else
struct NonTrivial {
  unsigned long long parts[4];
  NonTrivial(const NonTrivial &);
  NonTrivial &operator=(const NonTrivial &);
};
NonTrivial C4(NonTrivial a);
NonTrivial P4(NonTrivial a) {
  // expected-error@+1 {{'musttail' call cannot safely forward this indirect argument}}
  [[clang::musttail]] return C4(a);
}
#endif
