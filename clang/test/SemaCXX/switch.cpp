// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

void test() {
  bool x = true;
  switch (x) { // expected-warning {{bool}}
    case 0:
      break;
  }

  int n = 3;
  switch (n && true) { // expected-warning {{bool}}
    case 1:
      break;
  }
}

// PR5518
struct A { 
  operator int(); // expected-note{{conversion to integral type}}
};

void x() { 
  switch(A()) {
  }
}

enum E { e1, e2 };
struct B : A {
  operator E() const; // expected-note{{conversion to enumeration type}}
};

void x2() {
  switch (B()) { // expected-error{{multiple conversions}}
  }
}

struct C; // expected-note{{forward declaration}}

void x3(C &c) {
  switch (c) { // expected-error{{incomplete class type}}
  }
}

namespace test3 {
  enum En { A, B, C };
  template <En how> void foo() {
    int x = 0, y = 5;

    switch (how) { //expected-warning {{no case matching constant switch condition '2'}}
    case A: x *= y; break;
    case B: x += y; break;
    // No case for C, but it's okay because we have a constant condition.
    }
  }

  template void foo<A>();
  template void foo<B>();
  template void foo<C>(); //expected-note {{in instantiation}}
}

// PR9304 and rdar://9045501
void click_check_header_sizes() {
  switch (0 == 8) {  // expected-warning {{switch condition has boolean value}}
  case 0: ;
  }
}

void local_class(int n) {
  for (;;) switch (n) {
  case 0:
    struct S {
      void f() {
        case 1: // expected-error {{'case' statement not in switch statement}}
        break; // expected-error {{'break' statement not in loop or switch statement}}
        default: // expected-error {{'default' statement not in switch statement}}
        continue; // expected-error {{'continue' statement not in loop statement}}
      }
    };
    S().f();
    []{
      case 2: // expected-error {{'case' statement not in switch statement}}
      break; // expected-error {{'break' statement not in loop or switch statement}}
      default: // expected-error {{'default' statement not in switch statement}}
      continue; // expected-error {{'continue' statement not in loop statement}}
    }();
  }
}

namespace Conversion {
  struct S {
    explicit operator int(); // expected-note {{conversion}}
  };
  template<typename T> void f(T t) {
    switch (t) { // expected-error {{explicit conversion}}
    case 0:
      return;
    default:
      break;
    }
  }
  template void f(S); // expected-note {{instantiation of}}
}
