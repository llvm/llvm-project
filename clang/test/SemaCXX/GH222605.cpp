// RUN: %clang_cc1 -std=c++17 -fsyntax-only -verify %s
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %s
// RUN: %clang_cc1 -std=c++23 -fsyntax-only -verify %s

struct S {
  int *p;
  int *q;
};

struct Point {
  double x;
  double y;
  double z;
};

void test_recovered(S s, Point pt) {
  auto* [x, y] = s; // expected-error {{structured binding declaration cannot be declared with type 'auto *'; declared type must be 'auto' or reference to 'auto'}}
  x = nullptr;
  y = nullptr;

  const auto* [c1, c2] = s; // expected-error {{structured binding declaration cannot be declared with type 'const auto *'; declared type must be 'auto' or reference to 'auto'}}

  auto* [px, py, pz] = pt; // expected-error {{structured binding declaration cannot be declared with type 'auto *'; declared type must be 'auto' or reference to 'auto'}}
  double sum = px + py + pz;
  decltype(px) bad = "string"; // expected-error {{cannot initialize a variable of type 'decltype(px)' (aka 'double') with an lvalue of type 'const char[7]'}}
}

void test_unrecovered(S s) {
  auto** [p1, p2] = s; // expected-error {{structured binding declaration cannot be declared with type 'auto **'; declared type must be 'auto' or reference to 'auto'}} \
                       // expected-error {{variable '' with type 'auto **' has incompatible initializer of type 'S'}}

  auto* const [k1, k2] = s; // expected-error {{structured binding declaration cannot be declared with type 'auto *const'; declared type must be 'auto' or reference to 'auto'}} \
                            // expected-error {{variable '' with type 'auto *const' has incompatible initializer of type 'S'}}

  auto*& [r1, r2] = s; // expected-error {{structured binding declaration cannot be declared with type 'auto *&'; declared type must be 'auto' or reference to 'auto'}} \
                       // expected-error {{variable '' with type 'auto *&' has incompatible initializer of type 'S'}}

  auto S::*[m1, m2] = s; // expected-error {{structured binding declaration cannot be declared with type 'auto S::*'; declared type must be 'auto' or reference to 'auto'}} \
                         // expected-error {{variable '' with type 'auto S::*' has incompatible initializer of type 'S'}}
}

void test_pointer_object(S *sp) {
  auto* [a, b] = sp; // expected-error {{structured binding declaration cannot be declared with type 'auto *'; declared type must be 'auto' or reference to 'auto'}} \
                     // expected-error {{cannot bind non-class, non-array type 'S *'}}
}

template <typename T>
void test_template(T s) {
  auto* [x, y] = s; // expected-error {{structured binding declaration cannot be declared with type 'auto *'; declared type must be 'auto' or reference to 'auto'}}
  x = nullptr;
  y = nullptr;
}

void test_template_instantiation(S s) {
  test_template(s);
}
