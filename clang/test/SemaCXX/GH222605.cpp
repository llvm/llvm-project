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

void test_recovery(S s) {
  auto* [x, y] = s; // expected-error {{structured binding declaration cannot be declared with type 'auto *'; declared type must be 'auto' or reference to 'auto'}}
  x = nullptr;
  y = nullptr;

  auto*& [r1, r2] = s; // expected-error {{structured binding declaration cannot be declared with type 'auto *&'; declared type must be 'auto' or reference to 'auto'}}
  r1 = nullptr;
  r2 = nullptr;

  const auto* [c1, c2] = s; // expected-error {{structured binding declaration cannot be declared with type 'const auto *'; declared type must be 'auto' or reference to 'auto'}}

  const auto*& [cr1, cr2] = s; // expected-error {{structured binding declaration cannot be declared with type 'const auto *&'; declared type must be 'auto' or reference to 'auto'}}

  auto** [p1, p2] = s; // expected-error {{structured binding declaration cannot be declared with type 'auto **'; declared type must be 'auto' or reference to 'auto'}}
  p1 = nullptr;
  p2 = nullptr;
}

void test_aggregate(Point pt) {
  auto* [x, y, z] = pt; // expected-error {{structured binding declaration cannot be declared with type 'auto *'; declared type must be 'auto' or reference to 'auto'}}
  double sum = x + y + z;
}

void test_downstream_type_checking(Point pt) {
  auto* [x, y, z] = pt; // expected-error {{structured binding declaration cannot be declared with type 'auto *'; declared type must be 'auto' or reference to 'auto'}}
  decltype(x) bad = "string"; // expected-error {{cannot initialize a variable of type 'decltype(x)' (aka 'double') with an lvalue of type 'const char[7]'}}
}

void test_valid(S s, Point pt) {
  auto [a, b] = s;
  auto& [c, d] = s;
  const auto& [e, f] = s;
  auto&& [g, h] = S{};

  auto [x, y, z] = pt;
  const auto& [cx, cy, cz] = pt;
}

void test_other_errors(S s) {
  auto* p = s; // expected-error {{variable 'p' with type 'auto *' has incompatible initializer of type 'S'}}

  auto [a] = s; // expected-error {{type 'S' binds to 2 elements, but only 1 name was provided}}
  auto [b, c, d] = s; // expected-error {{type 'S' binds to 2 elements, but 3 names were provided}}

  int n = 0;
  auto [x, y] = n; // expected-error {{cannot bind non-class, non-array type 'int'}}
}

void test_pointer(S *sp) {
  auto* [a, b] = sp; // expected-error {{structured binding declaration cannot be declared with type 'auto *'; declared type must be 'auto' or reference to 'auto'}} \
                     // expected-error {{cannot bind non-class, non-array type 'S *'}}
  auto** [c, d] = &sp; // expected-error {{structured binding declaration cannot be declared with type 'auto **'; declared type must be 'auto' or reference to 'auto'}} \
                       // expected-error {{cannot bind non-class, non-array type 'S **'}}
}
