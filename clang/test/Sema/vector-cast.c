// RUN: %clang_cc1 -fsyntax-only %s -verify -Wvector-conversion

typedef long long t1 __attribute__ ((vector_size (8)));
typedef char t2 __attribute__ ((vector_size (16)));
typedef float t3 __attribute__ ((vector_size (16)));
typedef short s2 __attribute__ ((vector_size(4)));

typedef enum { Evalue = 0x10000 } E;

void f()
{  
  t1 v1;
  t2 v2;
  t3 v3;
  s2 v4;
  E e;

  e = (E)v4;
  v4 = (s2)e;
  
  v2 = (t2)v1; // expected-error {{invalid conversion between vector type \
't2' (vector of 16 'char' values) and 't1' (vector of 1 'long long' value) of different size}}
  v1 = (t1)v2; // expected-error {{invalid conversion between vector type \
't1' (vector of 1 'long long' value) and 't2' (vector of 16 'char' values) of different size}}
  v3 = (t3)v2;
  
  v1 = (t1)(char *)10; // expected-error {{invalid conversion between vector \
type 't1' (vector of 1 'long long' value) and scalar type 'char *'}}
  v1 = (t1)(long long)10;
  v1 = (t1)(short)10; // expected-error {{invalid conversion between vector \
type 't1' (vector of 1 'long long' value) and integer type 'short' of different size}}
  
  long long r1 = (long long)v1;
  short r2 = (short)v1; // expected-error {{invalid conversion between vector \
type 't1' (vector of 1 'long long' value) and integer type 'short' of different size}}
  char *r3 = (char *)v1; // expected-error {{invalid conversion between vector\
 type 't1' (vector of 1 'long long' value) and scalar type 'char *'}}
}


void f2(t2 X); // expected-note{{passing argument to parameter 'X' here}}

void f3(t3 Y) {
  f2(Y);  // expected-warning {{incompatible vector types passing 't3' (vector of 4 'float' values) to parameter of type 't2' (vector of 16 'char' values)}}
}

typedef float float2 __attribute__ ((vector_size (8)));

void f4() {
  float2 f2;
  double d;
  f2 += d;
  d += f2;
}

// rdar://15931426
// Don't permit a lax conversion to and from a pointer type.
typedef short short_sizeof_pointer __attribute__((vector_size(sizeof(void*))));
void f5() {
  short_sizeof_pointer v;
  void *ptr;
  v = ptr; // expected-error-re {{assigning to 'short_sizeof_pointer' (vector of {{[0-9]+}} 'short' values) from incompatible type 'void *'}}
  ptr = v; // expected-error {{assigning to 'void *' from incompatible type 'short_sizeof_pointer'}}
}
