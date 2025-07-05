// RUN: %clang_cc1 -fsyntax-only -verify %s

namespace test_deleted_ctor_note {
struct A {
  int a;
  A() = delete; // expected-note {{'A' has been explicitly marked deleted here}}
  A(int a_) : a(a_) { }
};

struct B {
  A a1, a2, a3; // expected-note {{default constructed field 'a2' declared here}}
  B(int a_) : a1(a_), a3(a_) { } // expected-error{{call to deleted constructor of 'A'}}
};
}
