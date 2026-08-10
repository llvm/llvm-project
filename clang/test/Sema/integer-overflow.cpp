// RUN: %clang_cc1 %s -Wno-unused-value -verify -fsyntax-only

namespace GH58944 {
struct A {
  A(unsigned long) ;
};

A a(1024 * 1024 * 1024 * 1024 * 1024ull); // expected-warning {{overflow in expression; result is 0 with type 'int'}}

void f() {
  new int[1024 * 1024 * 1024 * 1024 * 1024ull]; // expected-warning {{overflow in expression; result is 0 with type 'int'}}

  int arr[]{1,2,3};
  arr[1024 * 1024 * 1024 * 1024 * 1024ull]; // expected-warning {{overflow in expression; result is 0 with type 'int'}}

  (int){1024 * 1024 * 1024 * 1024 * 1024}; // expected-warning {{overflow in expression; result is 0 with type 'int'}}
}
}
