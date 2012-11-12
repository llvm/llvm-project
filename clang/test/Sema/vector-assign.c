// RUN: %clang_cc1 %s -verify -fsyntax-only -Wvector-conversion
typedef unsigned int v2u __attribute__ ((vector_size (8)));
typedef signed int v2s __attribute__ ((vector_size (8)));
typedef signed int v1s __attribute__ ((vector_size (4)));
typedef float v2f __attribute__ ((vector_size(8)));
typedef signed short v4ss __attribute__ ((vector_size (8)));

void test1() {
  v2s v1;
  v2u v2;
  v1s v3;
  v2f v4;
  v4ss v5;
  
  v1 = v2; // expected-warning {{incompatible vector types assigning to 'v2s' from 'v2u'}}
  v1 = v3; // expected-error {{assigning to 'v2s' from incompatible type 'v1s'}}
  v1 = v4; // expected-warning {{incompatible vector types assigning to 'v2s' from 'v2f'}}
  v1 = v5; // expected-warning {{incompatible vector types assigning to 'v2s' from 'v4ss'}}
  
  v2 = v1; // expected-warning {{incompatible vector types assigning to 'v2u' from 'v2s'}}
  v2 = v3; // expected-error {{assigning to 'v2u' from incompatible type 'v1s'}}
  v2 = v4; // expected-warning {{incompatible vector types assigning to 'v2u' from 'v2f'}}
  v2 = v5; // expected-warning {{incompatible vector types assigning to 'v2u' from 'v4ss'}}
  
  v3 = v1; // expected-error {{assigning to 'v1s' from incompatible type 'v2s'}}
  v3 = v2; // expected-error {{assigning to 'v1s' from incompatible type 'v2u'}}
  v3 = v4; // expected-error {{assigning to 'v1s' from incompatible type 'v2f'}}
  v3 = v5; // expected-error {{assigning to 'v1s' from incompatible type 'v4ss'}}
  
  v4 = v1; // expected-warning {{incompatible vector types assigning to 'v2f' from 'v2s'}}
  v4 = v2; // expected-warning {{incompatible vector types assigning to 'v2f' from 'v2u'}}
  v4 = v3; // expected-error {{assigning to 'v2f' from incompatible type 'v1s'}}
  v4 = v5; // expected-warning {{incompatible vector types assigning to 'v2f' from 'v4ss'}}
  
  v5 = v1; // expected-warning {{incompatible vector types assigning to 'v4ss' from 'v2s'}}
  v5 = v2; // expected-warning {{incompatible vector types assigning to 'v4ss' from 'v2u'}}
  v5 = v3; // expected-error {{assigning to 'v4ss' from incompatible type 'v1s'}}
  v5 = v4; // expected-warning {{incompatible vector types assigning to 'v4ss' from 'v2f'}}
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
