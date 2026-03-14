// RUN: %clang_cc1 -fsyntax-only -verify %s
// PR4103: Make sure we have a location for the error
class A { 
  float a(int *); // expected-note{{passing argument to parameter here}}
  int b(); 
};
int A::b() { return a(a((int*)0)); } // expected-error {{cannot initialize a parameter of type 'int *' with an rvalue of type 'float'}}

