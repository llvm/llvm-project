// RUN: %clang_cc1 -std=c++20 -verify %s

template <class T>
struct A {
  void primary();
};

template <class T>
  requires requires(T &t) { requires sizeof(t) > 4; }
struct A<T> {
  void specialization1();
};

template <class T>
  requires requires(T &t) { requires sizeof(t) > 8; }
struct A<T> {
  void specialization2();
};

int main() {
  A<char>().primary();
  A<char[5]>().specialization1();
  A<char[16]>(); // expected-error {{ambiguous partial specialization}}
                 // expected-note@10 {{partial specialization matches [with T = char[16]}}
                 // expected-note@16 {{partial specialization matches [with T = char[16]}}
}

// Check error messages when no overload with constraints matches.
template <class T>
void foo()
  requires requires(T &t) { requires sizeof(t) < 4; }
{}

template <class T>
void foo()
  requires requires(T &t) { requires sizeof(t) > 4; }
{}

template <class T>
void foo()
  requires requires(T &t) { requires sizeof(t) > 8; }
{}

void test() {
  foo<char[4]>();
  // expected-error@-1 {{no matching function for call to 'foo'}}
  // expected-note@30 {{candidate template ignored: constraints not satisfied}}
  // expected-note@31 {{because 'sizeof (t) < 4' (4 < 4) evaluated to false}}
  // expected-note@35 {{candidate template ignored: constraints not satisfied}}
  // expected-note@36 {{because 'sizeof (t) > 4' (4 > 4) evaluated to false}}
  // expected-note@40 {{candidate template ignored: constraints not satisfied}}
  // expected-note@41 {{because 'sizeof (t) > 8' (4 > 8) evaluated to false}}

  foo<char[16]>();
  // expected-error@-1 {{call to 'foo' is ambiguous}}
  // expected-note@35 {{candidate function}}
  // expected-note@40 {{candidate function}}
}
