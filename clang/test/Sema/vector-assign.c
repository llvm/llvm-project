// RUN: %clang_cc1 %s -verify -fsyntax-only -Wvector-conversion
typedef unsigned int v2u __attribute__ ((vector_size (8)));
typedef signed int v2s __attribute__ ((vector_size (8)));
typedef signed int v1s __attribute__ ((vector_size (4)));
typedef float v2f __attribute__ ((vector_size(8)));
typedef signed short v4ss __attribute__ ((vector_size (8)));

void test1(void) {
  v2s v1;
  v2u v2;
  v1s v3;
  v2f v4;
  v4ss v5;
  
  v1 = v2; // expected-warning {{incompatible vector types assigning to 'v2s' (vector of 2 'int' values) from 'v2u' (vector of 2 'unsigned int' values)}}
  v1 = v3; // expected-error {{assigning to 'v2s' (vector of 2 'int' values) from incompatible type 'v1s' (vector of 1 'int' value)}}
  v1 = v4; // expected-warning {{incompatible vector types assigning to 'v2s' (vector of 2 'int' values) from 'v2f' (vector of 2 'float' values)}}
  v1 = v5; // expected-warning {{incompatible vector types assigning to 'v2s' (vector of 2 'int' values) from 'v4ss' (vector of 4 'short' values)}}
  
  v2 = v1; // expected-warning {{incompatible vector types assigning to 'v2u' (vector of 2 'unsigned int' values) from 'v2s' (vector of 2 'int' values)}}
  v2 = v3; // expected-error {{assigning to 'v2u' (vector of 2 'unsigned int' values) from incompatible type 'v1s' (vector of 1 'int' value)}}
  v2 = v4; // expected-warning {{incompatible vector types assigning to 'v2u' (vector of 2 'unsigned int' values) from 'v2f' (vector of 2 'float' values)}}
  v2 = v5; // expected-warning {{incompatible vector types assigning to 'v2u' (vector of 2 'unsigned int' values) from 'v4ss' (vector of 4 'short' values)}}
  
  v3 = v1; // expected-error {{assigning to 'v1s' (vector of 1 'int' value) from incompatible type 'v2s' (vector of 2 'int' values)}}
  v3 = v2; // expected-error {{assigning to 'v1s' (vector of 1 'int' value) from incompatible type 'v2u' (vector of 2 'unsigned int' values)}}
  v3 = v4; // expected-error {{assigning to 'v1s' (vector of 1 'int' value) from incompatible type 'v2f' (vector of 2 'float' values)}}
  v3 = v5; // expected-error {{assigning to 'v1s' (vector of 1 'int' value) from incompatible type 'v4ss'}}
  
  v4 = v1; // expected-warning {{incompatible vector types assigning to 'v2f' (vector of 2 'float' values) from 'v2s' (vector of 2 'int' values)}}
  v4 = v2; // expected-warning {{incompatible vector types assigning to 'v2f' (vector of 2 'float' values) from 'v2u' (vector of 2 'unsigned int' values)}}
  v4 = v3; // expected-error {{assigning to 'v2f' (vector of 2 'float' values) from incompatible type 'v1s' (vector of 1 'int' value)}}
  v4 = v5; // expected-warning {{incompatible vector types assigning to 'v2f' (vector of 2 'float' values) from 'v4ss' (vector of 4 'short' values)}}
  
  v5 = v1; // expected-warning {{incompatible vector types assigning to 'v4ss' (vector of 4 'short' values) from 'v2s' (vector of 2 'int' values)}}
  v5 = v2; // expected-warning {{incompatible vector types assigning to 'v4ss' (vector of 4 'short' values) from 'v2u' (vector of 2 'unsigned int' values)}}
  v5 = v3; // expected-error {{assigning to 'v4ss' (vector of 4 'short' values) from incompatible type 'v1s' (vector of 1 'int' value)}}
  v5 = v4; // expected-warning {{incompatible vector types assigning to 'v4ss' (vector of 4 'short' values) from 'v2f'}}
}

// PR2263
float test2(__attribute__((vector_size(16))) float a, int b) {
   return a[b];
}

// PR4838
typedef long long __attribute__((__vector_size__(2 * sizeof(long long))))
longlongvec;

void test3a(longlongvec *); // expected-note{{passing argument to parameter here}}
void test3(const unsigned *src) {
  test3a(src);  // expected-warning {{incompatible pointer types passing 'const unsigned int *' to parameter of type 'longlongvec *'}}
}
