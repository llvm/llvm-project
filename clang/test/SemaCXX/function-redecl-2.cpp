// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

namespace redecl_in_templ {
template<typename T> void redecl_in_templ() {
  extern void func_1();  // expected-note  {{previous declaration is here}}
  extern int  func_1();  // expected-error {{functions that differ only in their return type cannot be overloaded}}
}

void g();
constexpr void (*p)() = g;

template<bool> struct X {};
template<> struct X<true> { typedef int type; };

template<typename T> void f() {
  extern void g();
  X<&g == p>::type n;
}
}
