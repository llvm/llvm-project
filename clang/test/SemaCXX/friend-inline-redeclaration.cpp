// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++23 %s

// Friend inline re-declaration, where the parameter type is a SAME class
struct A;

int foo(A&) { return 1; }
double compute(A&&) { return 3.14; }
char process(A&, char) { return 'x'; }
long action(long, A&) { return 42; }

struct A {
  friend inline int foo(A&);
  friend inline double compute(A&&);
  friend inline char process(A&, char);
  friend inline long action(long, A&);
};

// Friend inline re-declaration, but the parameter type is a DIFFERENT class,
// not the enclosing one. The exception should NOT apply.
struct Other {};
struct Owner;

int bar(Other&) { return 0; } // expected-note {{previous definition is here}}

struct Owner {
  friend inline int bar(Other&); // expected-error {{inline declaration of 'bar' follows non-inline definition}}
};

// Friend inline re-declaration with no parameters at all.
// The exception requires at least one parameter of the enclosing class type.
int init() { return 0; } // expected-note {{previous definition is here}}

struct Engine {
  friend inline int init(); // expected-error {{inline declaration of 'init' follows non-inline definition}}
};
