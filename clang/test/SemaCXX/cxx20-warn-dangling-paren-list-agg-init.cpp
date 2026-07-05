// RUN: %clang_cc1 -verify -std=c++20 %s -fsyntax-only

namespace std {
template <class T> struct remove_reference { typedef T type; };
template <class T> struct remove_reference<T&> { typedef T type; };
template <class T> struct remove_reference<T&&> { typedef T type; };

template <class T> typename remove_reference<T>::type &&move(T &&t);
} // namespace std

// dcl.init 16.6.2.2
struct A {
  int a;
  int&& r;
};

int f();
int n = 10;

A a1{1, f()}; // OK, lifetime is extended for direct-list-initialization
// well-formed, but dangling reference
A a2(1, f()); // expected-warning {{temporary whose address is used as value}}
// well-formed, but dangling reference
A a4(1.0, 1); // expected-warning {{temporary whose address is used as value}}
A a5(1.0, std::move(n));  // OK



struct B {
  const int& r;
};
B test(int local) {
  return B(1); // expected-warning {{returning address}}
  return B(local); // expected-warning {{address of stack memory}}
}

void f(B b);
void test2(int local) {
  // No diagnostic on the following cases where both the aggregate object and
  // temporary end at the end of the full expression.
  f(B(1));
  f(B(local));
}

// Test nested struct.
struct C {
  B b;
};

struct D {
  C c;
};

C c1(B(
  1  // expected-warning {{temporary whose address is used as value}}
)); 
D d1(C(B(
  1  // expected-warning {{temporary whose address is used as value}}
)));
