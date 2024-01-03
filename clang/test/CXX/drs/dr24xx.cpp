// RUN: %clang_cc1 -std=c++98 %s -verify=expected
// RUN: %clang_cc1 -std=c++11 %s -verify=expected
// RUN: %clang_cc1 -std=c++14 %s -verify=expected
// RUN: %clang_cc1 -std=c++17 %s -verify=expected,since-cxx17
// RUN: %clang_cc1 -std=c++20 %s -verify=expected,since-cxx17
// RUN: %clang_cc1 -std=c++23 %s -verify=expected,since-cxx17
// RUN: %clang_cc1 -std=c++2c %s -verify=expected,since-cxx17

#if __cplusplus <= 201402L
// expected-no-diagnostics
#endif

namespace dr2406 { // dr2406: 5
#if __cplusplus >= 201703L
void fallthrough(int n) {
  void g(), h(), i();
  switch (n) {
  case 1:
  case 2:
    g();
    [[fallthrough]];
  case 3: // warning on fallthrough discouraged
    do {
      [[fallthrough]];
      // since-cxx17-error@-1 {{fallthrough annotation does not directly precede switch label}}
    } while (false);
  case 6:
    do {
      [[fallthrough]];
      // since-cxx17-error@-1 {{fallthrough annotation does not directly precede switch label}}
    } while (n);
  case 7:
    while (false) {
      [[fallthrough]];
      // since-cxx17-error@-1 {{fallthrough annotation does not directly precede switch label}}
    }
  case 5:
    h();
  case 4: // implementation may warn on fallthrough
    i();
    [[fallthrough]];
    // since-cxx17-error@-1 {{fallthrough annotation does not directly precede switch label}}
  }
}
#endif
}

namespace dr2450 { // dr2450: 18 drafting
#if __cplusplus >= 202302L
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
#if __cplusplus >= 202302L
struct A {
  constexpr A(float) {}
};
template<A> struct X {};
X<1> x;
#endif
}
