// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

// Make sure we don't run off the end of the stream when parsing a deferred
// initializer.
int a; // expected-note {{previous}}
struct S {
  int n = 4 + ; // expected-error {{expected expression}}
} a; // expected-error {{redefinition}}

// Make sure we use all of the tokens.
struct T {
  int a = 1 // expected-error {{expected ';' at end of declaration list}}
  int b = 2;
  int c = b; // expected-error {{undeclared identifier}}
};

// Test recovery for bad constructor initializers

struct R1 {
  int a;
  R1() : a {}
}; // expected-error {{expected '{' or ','}}

// Test correct parsing.

struct V1 {
  int a, b;
  V1() : a(), b{} {}
};

template <typename, typename> struct T1 { enum {V};};
template <int, int> struct T2 { enum {V};};
struct A {
  T1<int, int> a1 = T1<int, int>(), *a2 = new T1<int,int>;
  T2<0,0> b1 = T2<0,0>(), b2 = T2<0,0>(), b3;
  bool c1 = 1 < 2, c2 = 2 < 1, c3 = false;
  bool d1 = T1<int, T1<int, int>>::V < 3, d2;
  T1<int, int()> e = T1<int, int()>();
};

struct PR19993 {
  static int n = delete; // expected-error {{only functions can have deleted definitions}}
};
