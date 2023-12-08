// RUN: %clang_cc1 -std=c++1y -verify %s

class [[deprecated]] C {}; // expected-note {{'C' has been explicitly marked deprecated here}}
C c; // expected-warning {{'C' is deprecated}}

typedef int t [[deprecated]]; // expected-note {{'t' has been explicitly marked deprecated here}}
t x = 42; // expected-warning {{'t' is deprecated}}

[[deprecated]] int old = 42; // expected-note {{'old' has been explicitly marked deprecated here}}
int use = old; // expected-warning {{'old' is deprecated}}

struct S { [[deprecated]] int member = 42; } s; // expected-note {{'member' has been explicitly marked deprecated here}}
int use2 = s.member; // expected-warning {{'member' is deprecated}}

[[deprecated]] int f() { return 42; } // expected-note {{'f' has been explicitly marked deprecated here}}
int use3 = f(); // expected-warning {{'f' is deprecated}}

enum [[deprecated]] e { E }; // expected-note {{'e' has been explicitly marked deprecated here}}
e my_enum; // expected-warning {{'e' is deprecated}}

template <typename T> class X {};
template <> class [[deprecated]] X<int> {}; // expected-note {{'X<int>' has been explicitly marked deprecated here}}
X<char> x1;
X<int> x2; // expected-warning {{'X<int>' is deprecated}}

template <typename T> class [[deprecated]] X2 {}; //expected-note {{'X2<char>' has been explicitly marked deprecated here}}
template <> class X2<int> {};
X2<char> x3; // expected-warning {{'X2<char>' is deprecated}}
X2<int> x4; // No warning, the specialization removes it.

template <typename T> class [[deprecated]] X3; //expected-note {{'X3<char>' has been explicitly marked deprecated here}}
template <> class X3<int>;
X3<char> *x5; // expected-warning {{'X3<char>' is deprecated}}
X3<int> *x6; // No warning, the specialization removes it.

template <typename T> struct A;
A<int> *p;
template <typename T> struct [[deprecated]] A;//expected-note {{'A<int>' has been explicitly marked deprecated here}} expected-note {{'A<float>' has been explicitly marked deprecated here}}
A<int> *q; // expected-warning {{'A<int>' is deprecated}}
A<float> *r; // expected-warning {{'A<float>' is deprecated}}

template <typename T> struct B;
B<int> *p2;
template <typename T> struct [[deprecated]] B;//expected-note {{'B<int>' has been explicitly marked deprecated here}} expected-note {{'B<float>' has been explicitly marked deprecated here}}
B<int> *q2; // expected-warning {{'B<int>' is deprecated}}
B<float> *r2; // expected-warning {{'B<float>' is deprecated}}

template <typename T> 
T some_func(T t) {
  struct [[deprecated]] FunS{}; // expected-note {{'FunS' has been explicitly marked deprecated here}}
  FunS f;// expected-warning {{'FunS' is deprecated}}

}

template <typename T>
[[deprecated]]T some_func2(T t) {
  struct FunS2{};
  FunS2 f;// No warning, entire function is deprecated, so usage here should be fine.

}
