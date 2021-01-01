// RUN: %clang_cc1 -std=c++20 -fsyntax-only -fcxx-exceptions -verify=expected %s
// RUN: %clang_cc1 -std=c++17 -fsyntax-only -fcxx-exceptions -verify=expected %s
// RUN: %clang_cc1 -std=c++14 -fsyntax-only -fcxx-exceptions -verify=expected %s
// RUN: %clang_cc1 -std=c++11 -fsyntax-only -fcxx-exceptions -verify=expected %s

namespace test_delete_function {
struct A1 {
  A1();
  A1(const A1 &);
  A1(A1 &&) = delete; // expected-note {{'A1' has been explicitly marked deleted here}}
};
A1 test1() {
  A1 a;
  return a; // expected-error {{call to deleted constructor of 'test_delete_function::A1'}}
}

struct A2 {
  A2();
  A2(const A2 &);

private:
  A2(A2 &&); // expected-note {{declared private here}}
};
A2 test2() {
  A2 a;
  return a; // expected-error {{calling a private constructor of class 'test_delete_function::A2'}}
}

struct C {};

struct B1 {
  B1(C &);
  B1(C &&) = delete; // expected-note {{'B1' has been explicitly marked deleted here}}
};
B1 test3() {
  C c;
  return c; // expected-error {{conversion function from 'test_delete_function::C' to 'test_delete_function::B1' invokes a deleted function}}
}

struct B2 {
  B2(C &);

private:
  B2(C &&); // expected-note {{declared private here}}
};
B2 test4() {
  C c;
  return c; // expected-error {{calling a private constructor of class 'test_delete_function::B2'}}
}
} // namespace test_delete_function
