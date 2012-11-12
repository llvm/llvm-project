// RUN: %clang_cc1 -fcxx-exceptions -fexceptions -fsyntax-only -verify %s -std=c++11

struct S {
  virtual ~S();

  void g() throw (auto(*)()->int);

  // Note, this is not permitted: conversion-declarator cannot have a trailing return type.
  // FIXME: don't issue the second diagnostic for this.
  operator auto(*)()->int(); // expected-error{{'auto' not allowed here}} expected-error {{C++ requires a type specifier}}
};

typedef auto Fun(int a) -> decltype(a + a);
typedef auto (*PFun)(int a) -> decltype(a + a);

void g(auto (*f)() -> int) {
  try { }
  catch (auto (&f)() -> int) { }
  catch (auto (*const f[10])() -> int) { }
}

namespace std {
  class type_info;
}

template<typename T> struct U {};

void j() {
  (void)typeid(auto(*)()->void);
  (void)sizeof(auto(*)()->void);
  (void)__alignof(auto(*)()->void);

  U<auto(*)()->void> v;

  int n;
  (void)static_cast<auto(*)()->void>(&j);
  auto p = reinterpret_cast<auto(*)()->int>(&j);
  (void)const_cast<auto(**)()->int>(&p);
  (void)(auto(*)()->void)(&j);
}

template <auto (*f)() -> void = &j> class C { };
struct F : auto(*)()->int {}; // expected-error{{expected class name}}
template<typename T = auto(*)()->int> struct G { };

int g();
auto (*h)() -> auto = &g; // expected-error{{'auto' not allowed in function return type}}
auto (*i)() = &g; // ok; auto deduced as int.
auto (*k)() -> int = i; // ok; no deduction.
