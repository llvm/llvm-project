// RUN: %clang_cc1 -fsyntax-only -Wdangling-field -verify -std=c++11 %s

struct X {
  X(int);
};
struct Y {
  operator X*();
  operator X&();
};

struct S {
  int &x, *y;  // expected-note {{reference member declared here}} \
               // expected-note {{pointer member declared here}}
  S(int i)
    : x(i),    // expected-warning {{binding reference member 'x' to stack allocated parameter 'i'}}
      y(&i) {} // expected-warning {{initializing pointer member 'y' with the stack address of parameter 'i'}}
  S(int &i) : x(i), y(&i) {} // no-warning: reference parameter
  S(int *i) : x(*i), y(i) {} // no-warning: pointer parameter
};

struct S2 {
  const X &x; // expected-note {{reference member declared here}}
  S2(int i) : x(i) {} // expected-warning {{binding reference member 'x' to a temporary}}
};

struct S3 {
  X &x1, *x2;
  S3(Y y) : x1(y), x2(y) {} // no-warning: conversion operator
};

template <typename T> struct S4 {
  T x; // expected-note {{reference member declared here}}
  S4(int i) : x(i) {} // expected-warning {{binding reference member 'x' to stack allocated parameter 'i'}}
};

template struct S4<int>; // no warning from this instantiation
template struct S4<int&>; // expected-note {{in instantiation}}

struct S5 {
  const X &x; // expected-note {{here}}
};
S5 s5 = { 0 }; // ok, lifetime-extended

struct S6 {
  S5 s5; // expected-note {{here}}
  S6() : s5 { 0 } {} // expected-warning {{binding reference subobject of member 's5' to a temporary}}
};

struct S7 : S5 {
  S7() : S5 { 0 } {} // expected-warning {{binding reference member 'x' to a temporary}}
};
