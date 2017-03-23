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

template <typename T> class [[deprecated]] X2 {};
template <> class X2<int> {};
X2<char> x3; // FIXME: no warning!
X2<int> x4;
