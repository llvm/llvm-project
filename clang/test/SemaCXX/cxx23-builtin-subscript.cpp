// RUN: %clang_cc1 -std=c++23 -verify -fenable-matrix -fdeclspec %s

union U {};
struct S {};
enum E {};
enum class EC {};

using vec3 = int __attribute__((ext_vector_type(3)));
using mat3 = int __attribute__((matrix_type(3, 3)));

void f(int *p) {
  int a[4]{};
  vec3 v{};
  mat3 m;
  p[1, 2]; // expected-error {{built-in subscript operator for type 'int *' expects exactly one argument}}
  p[1, p]; // expected-error {{built-in subscript operator for type 'int *' expects exactly one argument}}
  1[1, 2]; // expected-error {{built-in subscript operator for type 'int' expects exactly one argument}}
  1[p, 2]; // expected-error {{built-in subscript operator for type 'int' expects exactly one argument}}
  1[p, 2]; // expected-error {{built-in subscript operator for type 'int' expects exactly one argument}}
  p[U{}, U{}]; // expected-error {{built-in subscript operator for type 'int *' expects exactly one argument}}
  p[E{}, 1]; // expected-error {{built-in subscript operator for type 'int *' expects exactly one argument}}
  p[EC{}, 1]; // expected-error {{built-in subscript operator for type 'int *' expects exactly one argument}}
  p[S{}, 1]; // expected-error {{built-in subscript operator for type 'int *' expects exactly one argument}}
  p[1u, 1l]; // expected-error {{built-in subscript operator for type 'int *' expects exactly one argument}}
  p[1, 2, 3]; // expected-error {{built-in subscript operator for type 'int *' expects exactly one argument}}
  a[1, 2]; // expected-error {{built-in subscript operator for type 'int[4]' expects exactly one argument}}
  a[1, p]; // expected-error {{built-in subscript operator for type 'int[4]' expects exactly one argument}}
  a[S{}, p]; // expected-error {{built-in subscript operator for type 'int[4]' expects exactly one argument}}
  a[1, 2, 3]; // expected-error {{built-in subscript operator for type 'int[4]' expects exactly one argument}}
  v[1, 2]; // expected-error {{built-in subscript operator for type 'vec3' (vector of 3 'int' values) expects exactly one argument}}
  v[1, p]; // expected-error {{built-in subscript operator for type 'vec3' (vector of 3 'int' values) expects exactly one argument}}
  v[S{}, p]; // expected-error {{built-in subscript operator for type 'vec3' (vector of 3 'int' values) expects exactly one argument}}
  v[1, 2, 3]; // expected-error {{built-in subscript operator for type 'vec3' (vector of 3 'int' values) expects exactly one argument}}
  E{}[2, 2]; // expected-error {{built-in subscript operator for type 'E' expects exactly one argument}}
  EC{}[2, 2]; // expected-error {{built-in subscript operator for type 'EC' expects exactly one argument}}

  m[1][3, 4]; // expected-error {{comma expressions are not allowed as indices in matrix subscript}}
  m[1][2, 3]; // expected-error {{comma expressions are not allowed as indices in matrix subscript}}
  m[1, 2][3, 4]; // expected-error {{comma expressions are not allowed as indices in matrix subscript}}

  U{}[2, 2]; // expected-error {{type 'U' does not provide a subscript operator}}
  S{}[2, 2]; // expected-error {{type 'S' does not provide a subscript operator}}
}

struct Prop {
  constexpr int get_two(int a, int b);
  constexpr int get_three(int a, int b, int c);
  constexpr void put_two(int a, int b, int c);
  constexpr void put_three(int a, int b, int c, int d);
  __declspec(property(get = get_two, put = put_two)) int two[][];
  __declspec(property(get = get_three, put = put_three)) int three[][][];
};

void f() {
  Prop p;
  p.two[1, 2]; // expected-error {{property subscript expects exactly one argument}}
  p.three[1, 2, 3]; // expected-error {{property subscript expects exactly one argument}}
  p.three[1, 2][3]; // expected-error {{property subscript expects exactly one argument}}
  p.three[1][2, 3]; // expected-error {{property subscript expects exactly one argument}}
  p.two[1, 2] = 3; // expected-error {{property subscript expects exactly one argument}}
  p.three[1, 2, 3] = 4; // expected-error {{property subscript expects exactly one argument}}
  p.three[1, 2][3] = 4; // expected-error {{property subscript expects exactly one argument}}
  p.three[1][2, 3] = 4; // expected-error {{property subscript expects exactly one argument}}
}
