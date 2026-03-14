// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++11

struct A { // expected-warning {{does not declare any constructor to initialize}}
     const int i; // expected-note{{const member 'i' will never be initialized}} expected-note {{implicitly deleted}}
     virtual void f() { } 
};

int main () {
      (void)A(); // expected-error {{call to implicitly-deleted default constructor}}
}
