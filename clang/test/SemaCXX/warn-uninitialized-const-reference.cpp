// RUN: %clang_cc1 -fsyntax-only -fcxx-exceptions -fexceptions -Wuninitialized-const-reference -verify %s

class A {
public:
  int i;
  A(){};
  A(const A &a){};
  A(int i) {}
  bool operator!=(const A &);
};

template <class T>
void ignore_template(const T &) {}
void ignore(const int &i) {}
void dont_ignore_non_empty(const int &i) { ; } // Calling this won't silence the warning for you
void dont_ignore_block(const int &i) {
  {}
} // Calling this won't silence the warning for you
void ignore_function_try_block_maybe_who_knows(const int &) try {
} catch (...) {
}
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

  int l;
  ignore_template(l); // This is a pattern to avoid "unused variable" warnings (e.g. boost::ignore_unused).
  ignore(l);
  dont_ignore_non_empty(l); // expected-warning {{variable 'l' is uninitialized when passed as a const reference argument here}}
  int l1;
  dont_ignore_block(l1); // expected-warning {{variable 'l1' is uninitialized when passed as a const reference argument here}}
  int l2;
  ignore_function_try_block_maybe_who_knows(l2); // expected-warning {{variable 'l2' is uninitialized when passed as a const reference argument here}}
}
