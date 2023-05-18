// RUN: %clang_cc1 -fsyntax-only -verify %s

// Split from function-template-specialization.cpp because the noreturn warning
// requires analysis-based warnings, which the other errors in that test case
// disable.

template <int N> void __attribute__((noreturn)) f3() { __builtin_unreachable(); }
template <> void f3<1>() { } // expected-warning {{function declared 'noreturn' should not return}}

#if __cplusplus >= 201103L
namespace PR21942 {
template <int>
struct A {
  void foo[[noreturn]]();
};

template <>
void A<0>::foo() {} // expected-warning{{function declared 'noreturn' should not return}}
}
#endif
