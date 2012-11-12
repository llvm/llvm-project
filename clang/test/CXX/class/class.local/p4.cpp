// RUN: %clang_cc1 -fsyntax-only -verify %s 

void f() {
  struct X {
    static int a; // expected-error {{static data member 'a' not allowed in local class 'X'}}
    int b;
    
    static void f() { }
  };
}
