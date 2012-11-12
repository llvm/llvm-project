// RUN: %clang_cc1 -fsyntax-only %s -verify -Wvector-conversion

typedef long long t1 __attribute__ ((vector_size (8)));
typedef char t2 __attribute__ ((vector_size (16)));
typedef float t3 __attribute__ ((vector_size (16)));

void f()
{  
  t1 v1;
  t2 v2;
  t3 v3;
  
  v2 = (t2)v1; // expected-error {{invalid conversion between vector type \
't2' and 't1' of different size}}
  v1 = (t1)v2; // expected-error {{invalid conversion between vector type \
't1' and 't2' of different size}}
  v3 = (t3)v2;
  
  v1 = (t1)(char *)10; // expected-error {{invalid conversion between vector \
type 't1' and scalar type 'char *'}}
  v1 = (t1)(long long)10;
  v1 = (t1)(short)10; // expected-error {{invalid conversion between vector \
type 't1' and integer type 'short' of different size}}
  
  long long r1 = (long long)v1;
  short r2 = (short)v1; // expected-error {{invalid conversion between vector \
type 't1' and integer type 'short' of different size}}
  char *r3 = (char *)v1; // expected-error {{invalid conversion between vector\
 type 't1' and scalar type 'char *'}}
}


void f2(t2 X); // expected-note{{passing argument to parameter 'X' here}}

void f3(t3 Y) {
  f2(Y);  // expected-warning {{incompatible vector types passing 't3' to parameter of type 't2'}}
}

