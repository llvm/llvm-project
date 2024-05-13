// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s 

void f() {
  int x = 3; // expected-note{{'x' declared here}}
  const int c = 2;
  struct C {
    int& x2 = x; // expected-error{{reference to local variable 'x' declared in enclosing function 'f'}}
    int cc = c;
  };
  (void)[]() mutable {
    int x = 3; // expected-note{{'x' declared here}}
    struct C {
      int& x2 = x; // expected-error{{reference to local variable 'x' declared in enclosing lambda expression}}
    }c; // expected-note {{required here}}
  };
  C(); // expected-note {{required here}}
}

