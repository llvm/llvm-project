// RUN: %clang_cc1 -std=c++20 -triple x86_64-linux-gnu -emit-llvm -o - %s | FileCheck %s

// The closure type of a lambda in the return type of a deduction guide becomes
// a template argument of the deduced class type. A deduction guide has no
// mangled name, so the closure type must be one of the scope enclosing the
// guide, and its call operator must have been instantiated along with the
// guide (`operator()(int)`, not `operator()(T)`).

namespace alias_rhs {

template <class T, class F> struct A {
  A(T, F f = {}) { f({}); }
};

template <class T> using AA = A<T, decltype([](T) {})>;

AA a{0};
// CHECK-LABEL: define {{.*}} @_ZN9alias_rhs1AIiNS_UliE{{[0-9]*}}_EEC2EiS1_(
// CHECK: call void @_ZNK9alias_rhsUliE{{[0-9]*}}_clEi(
// CHECK-LABEL: define {{.*}} @_ZNK9alias_rhsUliE{{[0-9]*}}_clEi(
} // namespace alias_rhs

namespace user_guide {

template <class T, class F> struct A {
  A(T, F f = {}) { f({}); }
};

template <class T> A(T) -> A<T, decltype([](T) {})>;

A a{0};
// CHECK-LABEL: define {{.*}} @_ZN10user_guide1AIiNS_UliE{{[0-9]*}}_EEC2EiS1_(
// CHECK: call void @_ZNK10user_guideUliE{{[0-9]*}}_clEi(
// CHECK-LABEL: define {{.*}} @_ZNK10user_guideUliE{{[0-9]*}}_clEi(
} // namespace user_guide
