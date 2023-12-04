// RUN: %clang_cc1 -std=c++20 %s -verify
// RUN: %clang_cc1 -std=c++23 %s -verify
// expected-no-diagnostics

namespace dr2450 { // dr2450: 18 drafting
#if __cplusplus > 202002L
struct S {int a;};
template <S s>
void f(){}

void test() {
f<{0}>();
f<{.a= 0}>();
}

#endif
}

namespace dr2459 { // dr2459: 18 drafting
#if __cplusplus > 202002L
struct A {
  constexpr A(float) {}
};
template<A> struct X {};
X<1> x;
#endif
}
