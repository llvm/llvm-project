// RUN: %clang_cc1 -std=c++11 %s -Wdeprecated -verify
// RUN: %clang_cc1 -std=c++11 %s -Wdeprecated-copy -verify

struct A {
    A& operator=(const A&) = default; // expected-warning {{definition of implicit copy constructor for 'A' is deprecated because it has a user-declared copy assignment operator}}
};

struct B {
    B& operator=(const B&) = delete; // expected-warning {{definition of implicit copy constructor for 'B' is deprecated because it has a user-declared copy assignment operator}}
};

void test() {
  A a1;
  A a2(a1); // expected-note {{implicit copy constructor for 'A' first required here}}

  B b1;
  B b2(b1); // expected-note {{implicit copy constructor for 'B' first required here}}
}

// PR45634
struct S {
    int i;
    S& operator=(const S&) = delete; // expected-warning {{definition of implicit copy constructor for 'S' is deprecated because it has a user-declared copy assignment operator}}
};

S test(const S &s) { return S(s); } // expected-note {{implicit copy constructor for 'S' first required here}}
