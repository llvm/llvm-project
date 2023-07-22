// RUN: %clang_cc1 %s -fenable-matrix -pedantic -verify -triple=x86_64-apple-darwin9

typedef float sx5x10_t __attribute__((matrix_type(5, 10)));
typedef int ix3x2_t __attribute__((matrix_type(3, 2)));
typedef double dx3x3 __attribute__((matrix_type(3, 3)));
typedef unsigned ix3x3 __attribute__((matrix_type(3, 3)));

// Verify that we can use the [[]] spelling of the attribute.
// We intentionally use the same type alias name to check that both versions
// define the same type.
typedef float [[clang::matrix_type(5, 10)]] sx5x10_t; // expected-warning {{[[]] attributes are a C2x extension}}
typedef int [[clang::matrix_type(3, 2)]] ix3x2_t; // expected-warning {{[[]] attributes are a C2x extension}}
[[clang::matrix_type(5, 10)]] typedef float sx5x10_t; // expected-warning {{[[]] attributes are a C2x extension}}
// expected-warning@-1 {{applying attribute 'matrix_type' to a declaration is deprecated; apply it to the type instead}}
[[clang::matrix_type(3, 2)]] typedef int ix3x2_t; // expected-warning {{[[]] attributes are a C2x extension}}
// expected-warning@-1 {{applying attribute 'matrix_type' to a declaration is deprecated; apply it to the type instead}}

// Attribute may not be used outside typedefs.
[[clang::matrix_type(3, 2)]] int ix3x2_var; // expected-warning {{[[]] attributes are a C2x extension}}
// expected-error@-1 {{'matrix_type' attribute only applies to typedefs}}
int [[clang::matrix_type(3, 2)]] ix3x2_var; // expected-warning {{[[]] attributes are a C2x extension}}
// expected-error@-1 {{'matrix_type' attribute only applies to typedefs}}

void transpose(sx5x10_t a, ix3x2_t b, dx3x3 c, int *d, int e) {
  a = __builtin_matrix_transpose(b);
  // expected-error@-1 {{assigning to 'sx5x10_t' (aka 'float __attribute__((matrix_type(5, 10)))') from incompatible type 'int __attribute__((matrix_type(2, 3)))'}}
  b = __builtin_matrix_transpose(b);
  // expected-error@-1 {{assigning to 'ix3x2_t' (aka 'int __attribute__((matrix_type(3, 2)))') from incompatible type 'int __attribute__((matrix_type(2, 3)))'}}
  __builtin_matrix_transpose(d);
  // expected-error@-1 {{1st argument must be a matrix}}
  __builtin_matrix_transpose(e);
  // expected-error@-1 {{1st argument must be a matrix}}
  __builtin_matrix_transpose("test");
  // expected-error@-1 {{1st argument must be a matrix}}

  ix3x3 m = __builtin_matrix_transpose(c);
  // expected-error@-1 {{initializing 'ix3x3' (aka 'unsigned int __attribute__((matrix_type(3, 3)))') with an expression of incompatible type 'double __attribute__((matrix_type(3, 3)))'}}
}

struct Foo {
  unsigned x;
};

void column_major_load(float *p1, int *p2, _Bool *p3, struct Foo *p4) {
  sx5x10_t a1 = __builtin_matrix_column_major_load(p1, 5, 11, 5);
  // expected-error@-1 {{initializing 'sx5x10_t' (aka 'float __attribute__((matrix_type(5, 10)))') with an expression of incompatible type 'float __attribute__((matrix_type(5, 11)))'}}
  sx5x10_t a2 = __builtin_matrix_column_major_load(p1, 5, 9, 5);
  // expected-error@-1 {{initializing 'sx5x10_t' (aka 'float __attribute__((matrix_type(5, 10)))') with an expression of incompatible type 'float __attribute__((matrix_type(5, 9)))'}}
  sx5x10_t a3 = __builtin_matrix_column_major_load(p1, 6, 10, 6);
  // expected-error@-1 {{initializing 'sx5x10_t' (aka 'float __attribute__((matrix_type(5, 10)))') with an expression of incompatible type 'float __attribute__((matrix_type(6, 10)))'}}
  sx5x10_t a4 = __builtin_matrix_column_major_load(p1, 4, 10, 4);
  // expected-error@-1 {{initializing 'sx5x10_t' (aka 'float __attribute__((matrix_type(5, 10)))') with an expression of incompatible type 'float __attribute__((matrix_type(4, 10)))'}}
  sx5x10_t a5 = __builtin_matrix_column_major_load(p1, 6, 9, 6);
  // expected-error@-1 {{initializing 'sx5x10_t' (aka 'float __attribute__((matrix_type(5, 10)))') with an expression of incompatible type 'float __attribute__((matrix_type(6, 9)))'}}
  sx5x10_t a6 = __builtin_matrix_column_major_load(p2, 5, 10, 6);
  // expected-error@-1 {{initializing 'sx5x10_t' (aka 'float __attribute__((matrix_type(5, 10)))') with an expression of incompatible type 'int __attribute__((matrix_type(5, 10)))'}}

  sx5x10_t a7 = __builtin_matrix_column_major_load(p1, 5, 10, 3);
  // expected-error@-1 {{stride must be greater or equal to the number of rows}}

  sx5x10_t a8 = __builtin_matrix_column_major_load(p3, 5, 10, 6);
  // expected-error@-1 {{1st argument must be a pointer to a valid matrix element type}}

  sx5x10_t a9 = __builtin_matrix_column_major_load(p4, 5, 10, 6);
  // expected-error@-1 {{1st argument must be a pointer to a valid matrix element type}}

  sx5x10_t a10 = __builtin_matrix_column_major_load(p1, 1ull << 21, 10, 6);
  // expected-error@-1 {{row dimension is outside the allowed range [1, 1048575}}
  sx5x10_t a11 = __builtin_matrix_column_major_load(p1, 10, 1ull << 21, 10);
  // expected-error@-1 {{column dimension is outside the allowed range [1, 1048575}}

  sx5x10_t a12 = __builtin_matrix_column_major_load(
      10,         // expected-error {{1st argument must be a pointer to a valid matrix element type}}
      1ull << 21, // expected-error {{row dimension is outside the allowed range [1, 1048575]}}
      1ull << 21, // expected-error {{column dimension is outside the allowed range [1, 1048575]}}
      "");        // expected-error {{incompatible pointer to integer conversion casting 'char[1]' to type 'unsigned long'}}

  sx5x10_t a13 = __builtin_matrix_column_major_load(
      10,  // expected-error {{1st argument must be a pointer to a valid matrix element type}}
      *p4, // expected-error {{casting 'struct Foo' to incompatible type 'unsigned long'}}
      "",  // expected-error {{column argument must be a constant unsigned integer expression}}
           // expected-error@-1 {{incompatible pointer to integer conversion casting 'char[1]' to type 'unsigned long'}}
      10);
}

void column_major_store(sx5x10_t *m1, ix3x2_t *m2, float *p1, int *p2, struct Foo *p3, const float *p4) {
  __builtin_matrix_column_major_store(*m1, p1, 1);
  // expected-error@-1 {{stride must be greater or equal to the number of rows}}
  __builtin_matrix_column_major_store(*m1, p1, 0);
  // expected-error@-1 {{stride must be greater or equal to the number of rows}}
  __builtin_matrix_column_major_store(*m1, p2, 10);
  // expected-error@-1 {{the pointee of the 2nd argument must match the element type of the 1st argument ('int' != 'float')}}
  __builtin_matrix_column_major_store(p1, p2, 10);
  // expected-error@-1 {{1st argument must be a matrix}}

  __builtin_matrix_column_major_store(
      "",   // expected-error {{1st argument must be a matrix}}
      10,   // expected-error {{2nd argument must be a pointer to a valid matrix element type}}
      *p3); // expected-error {{casting 'struct Foo' to incompatible type 'unsigned long'}}

  __builtin_matrix_column_major_store(
      *m1,
      10, // expected-error {{2nd argument must be a pointer to a valid matrix element type}}
      10);

  *m1 = __builtin_matrix_column_major_store(*m1, p1, 10);
  // expected-error@-1 {{assigning to 'sx5x10_t' (aka 'float __attribute__((matrix_type(5, 10)))') from incompatible type 'void'}}

  int x = __builtin_matrix_column_major_store(*m1, p1, 10);
  // expected-error@-1 {{initializing 'int' with an expression of incompatible type 'void'}}

  __builtin_matrix_column_major_store(*m1, p4, 20);
  // expected-error@-1 {{cannot store matrix to read-only pointer}}
}
