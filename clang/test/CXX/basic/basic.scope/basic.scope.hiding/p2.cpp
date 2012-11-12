// RUN: %clang_cc1 -fsyntax-only -verify %s

// rdar4641403
namespace N {
  struct X { // expected-note{{candidate found by name lookup}}
    float b;
  };
}

using namespace N;

typedef struct {
  int a;
} X; // expected-note{{candidate found by name lookup}}


struct Y { };
void Y(int) { }

void f() {
  X *x; // expected-error{{reference to 'X' is ambiguous}}
  Y(1); // okay
}

