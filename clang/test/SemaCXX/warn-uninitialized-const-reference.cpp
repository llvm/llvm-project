// RUN: %clang_cc1 -fsyntax-only -Wuninitialized-const-reference -verify %s

class A {
public:
  int i;
  A(){};
  A(const A &a){};
  A(int i) {}
  bool operator!=(const A &);
};

A const_ref_use_A(const A &a);
int const_ref_use(const int &i);
A const_use_A(const A a);
int const_use(const int i);

void f(int a) {
  int i;
  const_ref_use(i);             // expected-warning {{variable 'i' is uninitialized when passed as a const reference argument here}}
  int j = j + const_ref_use(j); // expected-warning {{variable 'j' is uninitialized when used within its own initialization}} expected-warning {{variable 'j' is uninitialized when passed as a const reference argument here}}
  A a1 = const_ref_use_A(a1);   // expected-warning {{variable 'a1' is uninitialized when passed as a const reference argument here}}
  int k = const_use(k);         // expected-warning {{variable 'k' is uninitialized when used within its own initialization}}
  A a2 = const_use_A(a2);       // expected-warning {{variable 'a2' is uninitialized when used within its own initialization}}
  A a3(const_ref_use_A(a3));    // expected-warning {{variable 'a3' is uninitialized when passed as a const reference argument here}}
  A a4 = a3 != a4;              // expected-warning {{variable 'a4' is uninitialized when used within its own initialization}} expected-warning {{variable 'a4' is uninitialized when passed as a const reference argument here}}
  int n = n;                    // expected-warning {{variable 'n' is uninitialized when used within its own initialization}}
  const_ref_use(n);

  A a5;
  const_ref_use_A(a5);

  int m;
  if (a < 42)
    m = 1;
  const_ref_use(m);
}
