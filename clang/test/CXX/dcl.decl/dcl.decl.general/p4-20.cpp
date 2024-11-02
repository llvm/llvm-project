// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %s

// expected-error@+2 {{non-templated function cannot have a requires clause}}
void f1(int a)
  requires true;
template <typename T>
auto f2(T a) -> bool
  requires true; // OK

// expected-error@+4 {{trailing return type must appear before trailing requires clause}}
template <typename T>
auto f3(T a)
  requires true
-> bool;

// expected-error@+2{{trailing requires clause can only be used when declaring a function}}
void (*pf)()
  requires true;

// expected-error@+1{{trailing requires clause can only be used when declaring a function}}
void g(int (*)() requires true);

// expected-error@+1{{expected expression}}
auto *p = new void(*)(char)
  requires true;
