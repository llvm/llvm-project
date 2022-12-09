// RUN: %clang_cc1 -triple x86_64-unknown-windows-msvc -std=c++2b -fsyntax-only %s -verify
// RUN: %clang_cc1 -triple x86_64-unknown-windows-itanium -std=c++2b -fsyntax-only %s -verify

struct S {
  void operator()();
};

struct T {
  virtual void operator()();
};

struct U {
  static void operator()();
};

struct V: virtual T {
  virtual void f();
};

struct W : virtual V {
  int i;
};

struct X {
  __UINTPTR_TYPE__ ptr;
  __UINTPTR_TYPE__ adj;
};

auto L = [](){};

void f() {
  auto pmf = &S::operator();

  __asm__ __volatile__ ("" : : "r"(&decltype(L)::operator()));
  // expected-error@-1{{cannot pass a pointer-to-member through register-constrained inline assembly parameter}}
  __asm__ __volatile__ ("" : : "r"(&S::operator()));
  // expected-error@-1{{cannot pass a pointer-to-member through register-constrained inline assembly parameter}}
  __asm__ __volatile__ ("" : : "r"(&T::operator()));
  // expected-error@-1{{cannot pass a pointer-to-member through register-constrained inline assembly parameter}}
  __asm__ __volatile__ ("" : : "r"(pmf));
  // expected-error@-1{{cannot pass a pointer-to-member through register-constrained inline assembly parameter}}
  __asm__ __volatile__ ("" : : "r"(&W::f));
  // expected-error@-1{{cannot pass a pointer-to-member through register-constrained inline assembly parameter}}
  __asm__ __volatile__ ("" : : "r"(&W::i));
  // expected-error@-1{{cannot pass a pointer-to-member through register-constrained inline assembly parameter}}

  __asm__ __volatile__ ("" : : "r"(X{0,0}));
  __asm__ __volatile__ ("" : : "r"(&U::operator()));
}
