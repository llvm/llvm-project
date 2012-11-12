// RUN: %clang_cc1 -fsyntax-only -verify %s

// PR6647
class C {
  // After the error, the rest of the tokens inside the default arg should be
  // skipped, avoiding a "expected ';' after class" after 'undecl'.
  void m(int x = undecl + 0); // expected-error {{use of undeclared identifier 'undecl'}}
};

typedef struct Inst {
  void m(int x=0);
} *InstPtr;

struct X {
  void f(int x = 1:); // expected-error {{unexpected end of default argument expression}}
};
