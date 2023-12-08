// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

struct s0; // expected-note {{forward declaration}}
char ar[sizeof(s0&)]; // expected-error {{invalid application of 'sizeof' to an incomplete type}}
void test() {
  char &r = ar[0];
  static_assert(alignof(r) == 1, "bad alignment"); // expected-warning {{GNU extension}}
  static_assert(alignof(char&) == 1, "bad alignment");
  static_assert(sizeof(r) == 1, "bad size");
  static_assert(sizeof(char&) == 1, "bad size");
}

int f();  // expected-note{{possible target for call}}
int f(int);  // expected-note{{possible target for call}}
void g() { 
  sizeof(&f); // expected-error{{reference to overloaded function could not be resolved; did you mean to call it with no arguments?}} \
  // expected-warning{{expression result unused}}
}

template<typename T> void f_template(); // expected-note{{possible target for call}}
template<typename T> void f_template(T*); // expected-note{{possible target for call}}
void rdar9659191() {
  (void)alignof(f_template<int>); // expected-error{{reference to overloaded function could not be resolved; did you mean to call it?}} expected-warning {{GNU extension}}
}
