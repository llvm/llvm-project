// RUN: %clang_cc1 -std=c++20 -ast-dump %s | FileCheck %s

namespace PR46029 {

template <int N>
void canary1();
template <int N>
void canary2();

template <int N>
struct A {
  void f() requires(N == 1) {
    static_assert(N == 1);
    canary1<N>();
  }
  void f() requires(N == 2) {
    static_assert(N == 2);
    canary2<N>();
  }
};

// This checks that `canary1<1>` and `canaray2<2>` are instantiated, thus
// indirectly validating that the correct candidates of `A::f` were really
// instantiated each time.
// The `static_assert`s validate we don't instantiate wrong candidates.

// CHECK:{{.*}}FunctionTemplateDecl {{.*}} canary1
// CHECK:      {{.*}}TemplateArgument integral
// CHECK-SAME: {{'1'$}}
template struct A<1>;

// CHECK:      {{.*}}FunctionTemplateDecl {{.*}} canary2
// CHECK:      {{.*}}TemplateArgument integral
// CHECK-SAME: {{'2'$}}
template struct A<2>;

template struct A<3>;
}
