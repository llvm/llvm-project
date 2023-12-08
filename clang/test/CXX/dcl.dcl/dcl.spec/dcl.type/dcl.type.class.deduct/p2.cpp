// RUN: %clang_cc1 -std=c++17 -verify=expected,cxx17 %s
// RUN: %clang_cc1 -std=c++20 -verify=expected,cxx20 %s

template<typename T> struct A { constexpr A(T) {} }; // expected-note 1+{{here}}

A a = 0;
A b(0);
A c = A(0);
A d = A{0};
auto *p = new A(0);
A *q = new A(0); // expected-error {{cannot form pointer to deduced class template specialization type}}

struct B {
  operator A() { // expected-error {{argument deduction not allowed in conversion function type}}
    return A(0);
  }
};

void f(A a); // expected-error {{argument deduction not allowed in function prototype}}
A f(); // expected-error {{argument deduction not allowed in function return type}}

template<A a> // cxx17-error {{argument deduction not allowed in template parameter}}
void f();
