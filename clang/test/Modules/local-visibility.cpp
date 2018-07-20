// RUN: %clang_cc1 -fsyntax-only -fmodules %s -verify
// RUN: %clang_cc1 -fsyntax-only %s -verify

// expected-no-diagnostics
template <typename Var>
struct S {
  template <unsigned N>
  struct Inner { };

  template <>
  struct Inner<0> { };
};

S<int>::Inner<1> I1;
S<int>::Inner<0> I0;
