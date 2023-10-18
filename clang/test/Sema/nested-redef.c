// RUN: %clang_cc1 -fsyntax-only -verify %s
struct X { // expected-note{{previous definition is here}}
  struct X { } x; // expected-error{{nested redefinition of 'X'}}
}; 

struct Y { };
void f(void) {
  struct Y { }; // okay: this is a different Y
}

struct T;
struct Z {
  struct T { int x; } t;
  struct U { int x; } u;
};

void f2(void) {
  struct T t;
  struct U u;
}

void f3(void) {
  struct G { // expected-note{{previous definition is here}}
    struct G {}; // expected-error{{nested redefinition of 'G'}}
  };
}
