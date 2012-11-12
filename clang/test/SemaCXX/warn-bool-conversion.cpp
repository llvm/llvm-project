// RUN: %clang_cc1 -fsyntax-only -verify %s

int* j = false; // expected-warning{{initialization of pointer of type 'int *' to null from a constant boolean expression}}

void foo(int* i, int *j=(false)) // expected-warning{{initialization of pointer of type 'int *' to null from a constant boolean expression}}
{
  foo(false); // expected-warning{{initialization of pointer of type 'int *' to null from a constant boolean expression}}
  foo((int*)false); // no-warning: explicit cast
  foo(0); // no-warning: not a bool, even though its convertible to bool

  foo(false == true); // expected-warning{{initialization of pointer of type 'int *' to null from a constant boolean expression}}
  foo((42 + 24) < 32); // expected-warning{{initialization of pointer of type 'int *' to null from a constant boolean expression}}

  const bool kFlag = false;
  foo(kFlag); // expected-warning{{initialization of pointer of type 'int *' to null from a constant boolean expression}}
}

char f(struct Undefined*);
double f(...);

// Ensure that when using false in metaprogramming machinery its conversion
// isn't flagged.
template <int N> struct S {};
S<sizeof(f(false))> s;
