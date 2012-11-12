// RUN: %clang_cc1 -fsyntax-only -std=c++11 -verify %s 

namespace test0 {
  struct A { // expected-note {{candidate function (the implicit copy assignment operator) not viable: 'this' argument has type 'const test0::A', but method is not marked const}} expected-note {{candidate function (the implicit move assignment operator) not viable: 'this' argument has type 'const test0::A', but method is not marked const}}
    A &operator=(void*); // expected-note {{candidate function not viable: 'this' argument has type 'const test0::A', but method is not marked const}}
  };

  void test(const A &a) {
    a = "help"; // expected-error {{no viable overloaded '='}}
  }
}
