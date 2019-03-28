// RUN: %clang_cc1 %s -pedantic -verify -fsyntax-only

// Test that 'private' is not parsed as an address space qualifier
// in regular C++ mode.

struct B {
  virtual ~B() // expected-error{{expected ';' at end of declaration list}}
private:
   void foo();
   private int* i; // expected-error{{expected ':'}}
};

void bar(private int*); //expected-error{{variable has incomplete type 'void'}} expected-error{{expected expression}}
