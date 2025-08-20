// RUN: %clang_cc1 -std=c99 %s -pedantic -verify -triple=x86_64-apple-darwin9

typedef double double2 __attribute__((ext_vector_type(2)));
typedef double double4 __attribute__((ext_vector_type(4)));
typedef float float2 __attribute__((ext_vector_type(2)));
typedef float float3 __attribute__((ext_vector_type(3)));
typedef float float4 __attribute__((ext_vector_type(4)));

typedef int int2 __attribute__((ext_vector_type(2)));
typedef int int3 __attribute__((ext_vector_type(3)));
typedef unsigned unsigned3 __attribute__((ext_vector_type(3)));
typedef unsigned unsigned4 __attribute__((ext_vector_type(4)));

struct Foo {
  char *p;
};

__attribute__((address_space(1))) int int_as_one;
typedef int bar;
bar b;

__attribute__((address_space(1))) float float_as_one;
typedef float waffle;
waffle waf;


void test_builtin_elementwise_abs(int i, double d, float4 v, int3 iv, unsigned u, unsigned4 uv) {
  struct Foo s = __builtin_elementwise_abs(i);
  // expected-error@-1 {{initializing 'struct Foo' with an expression of incompatible type 'int'}}

  i = __builtin_elementwise_abs();
  // expected-error@-1 {{too few arguments to function call, expected 1, have 0}}

  i = __builtin_elementwise_abs(i, i);
  // expected-error@-1 {{too many arguments to function call, expected 1, have 2}}

  i = __builtin_elementwise_abs(v);
  // expected-error@-1 {{assigning to 'int' from incompatible type 'float4' (vector of 4 'float' values)}}

  u = __builtin_elementwise_abs(u);
  // expected-error@-1 {{1st argument must be a scalar or vector of signed integer or floating-point types (was 'unsigned int')}}

  uv = __builtin_elementwise_abs(uv);
  // expected-error@-1 {{1st argument must be a scalar or vector of signed integer or floating-point types (was 'unsigned4' (vector of 4 'unsigned int' values))}}
}

void test_builtin_elementwise_add_sat(int i, short s, double d, float4 v, int3 iv, unsigned3 uv, int *p) {
  i = __builtin_elementwise_add_sat(p, d);
  // expected-error@-1 {{1st argument must be a scalar or vector of integer types (was 'int *')}}

  struct Foo foo = __builtin_elementwise_add_sat(i, i);
  // expected-error@-1 {{initializing 'struct Foo' with an expression of incompatible type 'int'}}

  i = __builtin_elementwise_add_sat(i);
  // expected-error@-1 {{too few arguments to function call, expected 2, have 1}}

  i = __builtin_elementwise_add_sat();
  // expected-error@-1 {{too few arguments to function call, expected 2, have 0}}

  i = __builtin_elementwise_add_sat(i, i, i);
  // expected-error@-1 {{too many arguments to function call, expected 2, have 3}}

  i = __builtin_elementwise_add_sat(v, iv);
  // expected-error@-1 {{1st argument must be a scalar or vector of integer types (was 'float4' (vector of 4 'float' values))}}

  i = __builtin_elementwise_add_sat(iv, v);
  // expected-error@-1 {{arguments are of different types ('int3' (vector of 3 'int' values) vs 'float4' (vector of 4 'float' values))}}

  i = __builtin_elementwise_add_sat(uv, iv);
  // expected-error@-1 {{arguments are of different types ('unsigned3' (vector of 3 'unsigned int' values) vs 'int3' (vector of 3 'int' values))}}

  v = __builtin_elementwise_add_sat(v, v);
  // expected-error@-1 {{1st argument must be a scalar or vector of integer types (was 'float4' (vector of 4 'float' values))}}

  s = __builtin_elementwise_add_sat(i, s);
  // expected-error@-1 {{arguments are of different types ('int' vs 'short')}}

  enum e { one,
           two };
  i = __builtin_elementwise_add_sat(one, two);

  i = __builtin_elementwise_add_sat(one, d);
  // expected-error@-1 {{arguments are of different types ('int' vs 'double')}}

  enum f { three };
  enum f x = __builtin_elementwise_add_sat(one, three);
  // expected-error@-1 {{invalid arithmetic between different enumeration types ('enum e' and 'enum f')}}

  _BitInt(32) ext; // expected-warning {{'_BitInt' in C17 and earlier is a Clang extension}}
  ext = __builtin_elementwise_add_sat(ext, ext);

  const int ci = 0;
  i = __builtin_elementwise_add_sat(ci, i);
  i = __builtin_elementwise_add_sat(i, ci);
  i = __builtin_elementwise_add_sat(ci, ci);

  i = __builtin_elementwise_add_sat(i, int_as_one); // ok (attributes don't match)?
  i = __builtin_elementwise_add_sat(i, b);          // ok (sugar doesn't match)?

  int A[10];
  A = __builtin_elementwise_add_sat(A, A);
  // expected-error@-1 {{1st argument must be a scalar or vector of integer types (was 'int *')}}

  int(ii);
  int j;
  j = __builtin_elementwise_add_sat(i, j);

  _Complex float c1, c2;
  c1 = __builtin_elementwise_add_sat(c1, c2);
  // expected-error@-1 {{1st argument must be a scalar or vector of integer types (was '_Complex float')}}
}

void test_builtin_elementwise_sub_sat(int i, short s, double d, float4 v, int3 iv, unsigned3 uv, int *p) {
  i = __builtin_elementwise_sub_sat(p, d);
  // expected-error@-1 {{1st argument must be a scalar or vector of integer types (was 'int *')}}

  struct Foo foo = __builtin_elementwise_sub_sat(i, i);
  // expected-error@-1 {{initializing 'struct Foo' with an expression of incompatible type 'int'}}

  i = __builtin_elementwise_sub_sat(i);
  // expected-error@-1 {{too few arguments to function call, expected 2, have 1}}

  i = __builtin_elementwise_sub_sat();
  // expected-error@-1 {{too few arguments to function call, expected 2, have 0}}

  i = __builtin_elementwise_sub_sat(i, i, i);
  // expected-error@-1 {{too many arguments to function call, expected 2, have 3}}

  i = __builtin_elementwise_sub_sat(v, iv);
  // expected-error@-1 {{1st argument must be a scalar or vector of integer types (was 'float4' (vector of 4 'float' values))}}

  i = __builtin_elementwise_sub_sat(iv, v);
  // expected-error@-1 {{arguments are of different types ('int3' (vector of 3 'int' values) vs 'float4' (vector of 4 'float' values))}}

  i = __builtin_elementwise_sub_sat(uv, iv);
  // expected-error@-1 {{arguments are of different types ('unsigned3' (vector of 3 'unsigned int' values) vs 'int3' (vector of 3 'int' values))}}

  v = __builtin_elementwise_sub_sat(v, v);
  // expected-error@-1 {{1st argument must be a scalar or vector of integer types (was 'float4' (vector of 4 'float' values))}}

  s = __builtin_elementwise_sub_sat(i, s);
  // expected-error@-1 {{arguments are of different types ('int' vs 'short')}}

  enum e { one,
           two };
  i = __builtin_elementwise_sub_sat(one, two);

  i = __builtin_elementwise_sub_sat(one, d);
  // expected-error@-1 {{arguments are of different types ('int' vs 'double')}}

  enum f { three };
  enum f x = __builtin_elementwise_sub_sat(one, three);
  // expected-error@-1 {{invalid arithmetic between different enumeration types ('enum e' and 'enum f')}}

  _BitInt(32) ext; // expected-warning {{'_BitInt' in C17 and earlier is a Clang extension}}
  ext = __builtin_elementwise_sub_sat(ext, ext);

  const int ci = 0;
  i = __builtin_elementwise_sub_sat(ci, i);
  i = __builtin_elementwise_sub_sat(i, ci);
  i = __builtin_elementwise_sub_sat(ci, ci);

  i = __builtin_elementwise_sub_sat(i, int_as_one); // ok (attributes don't match)?
  i = __builtin_elementwise_sub_sat(i, b);          // ok (sugar doesn't match)?

  int A[10];
  A = __builtin_elementwise_sub_sat(A, A);
  // expected-error@-1 {{1st argument must be a scalar or vector of integer types (was 'int *')}}

  int(ii);
  int j;
  j = __builtin_elementwise_sub_sat(i, j);

  _Complex float c1, c2;
  c1 = __builtin_elementwise_sub_sat(c1, c2);
  // expected-error@-1 {{1st argument must be a scalar or vector of integer types (was '_Complex float')}}
}

void test_builtin_elementwise_max(int i, short s, double d, float4 v, int3 iv, unsigned3 uv, int *p) {
  i = __builtin_elementwise_max(p, d);
  // expected-error@-1 {{1st argument must be a vector, integer or floating-point type (was 'int *')}}

  struct Foo foo = __builtin_elementwise_max(i, i);
  // expected-error@-1 {{initializing 'struct Foo' with an expression of incompatible type 'int'}}

  i = __builtin_elementwise_max(i);
  // expected-error@-1 {{too few arguments to function call, expected 2, have 1}}

  i = __builtin_elementwise_max();
  // expected-error@-1 {{too few arguments to function call, expected 2, have 0}}

  i = __builtin_elementwise_max(i, i, i);
  // expected-error@-1 {{too many arguments to function call, expected 2, have 3}}

  i = __builtin_elementwise_max(v, iv);
  // expected-error@-1 {{arguments are of different types ('float4' (vector of 4 'float' values) vs 'int3' (vector of 3 'int' values))}}

  i = __builtin_elementwise_max(uv, iv);
  // expected-error@-1 {{arguments are of different types ('unsigned3' (vector of 3 'unsigned int' values) vs 'int3' (vector of 3 'int' values))}}

  s = __builtin_elementwise_max(i, s);
  // expected-error@-1 {{arguments are of different types ('int' vs 'short')}}

  enum e { one,
           two };
  i = __builtin_elementwise_max(one, two);

  i = __builtin_elementwise_max(one, d);
  // expected-error@-1 {{arguments are of different types ('int' vs 'double')}}

  enum f { three };
  enum f x = __builtin_elementwise_max(one, three);
  // expected-error@-1 {{invalid arithmetic between different enumeration types ('enum e' and 'enum f')}}

  _BitInt(32) ext; // expected-warning {{'_BitInt' in C17 and earlier is a Clang extension}}
  ext = __builtin_elementwise_max(ext, ext);

  const int ci = 0;
  i = __builtin_elementwise_max(ci, i);
  i = __builtin_elementwise_max(i, ci);
  i = __builtin_elementwise_max(ci, ci);

  i = __builtin_elementwise_max(i, int_as_one); // ok (attributes don't match)?
  i = __builtin_elementwise_max(i, b);          // ok (sugar doesn't match)?

  int A[10];
  A = __builtin_elementwise_max(A, A);
  // expected-error@-1 {{1st argument must be a vector, integer or floating-point type (was 'int *')}}

  int(ii);
  int j;
  j = __builtin_elementwise_max(i, j);

  _Complex float c1, c2;
  c1 = __builtin_elementwise_max(c1, c2);
  // expected-error@-1 {{1st argument must be a vector, integer or floating-point type (was '_Complex float')}}
}

void test_builtin_elementwise_min(int i, short s, double d, float4 v, int3 iv, unsigned3 uv, int *p) {
  i = __builtin_elementwise_min(p, d);
  // expected-error@-1 {{1st argument must be a vector, integer or floating-point type (was 'int *')}}

  struct Foo foo = __builtin_elementwise_min(i, i);
  // expected-error@-1 {{initializing 'struct Foo' with an expression of incompatible type 'int'}}

  i = __builtin_elementwise_min(i);
  // expected-error@-1 {{too few arguments to function call, expected 2, have 1}}

  i = __builtin_elementwise_min();
  // expected-error@-1 {{too few arguments to function call, expected 2, have 0}}

  i = __builtin_elementwise_min(i, i, i);
  // expected-error@-1 {{too many arguments to function call, expected 2, have 3}}

  i = __builtin_elementwise_min(v, iv);
  // expected-error@-1 {{arguments are of different types ('float4' (vector of 4 'float' values) vs 'int3' (vector of 3 'int' values))}}

  i = __builtin_elementwise_min(uv, iv);
  // expected-error@-1 {{arguments are of different types ('unsigned3' (vector of 3 'unsigned int' values) vs 'int3' (vector of 3 'int' values))}}

  s = __builtin_elementwise_min(i, s);
  // expected-error@-1 {{arguments are of different types ('int' vs 'short')}}

  enum e { one,
           two };
  i = __builtin_elementwise_min(one, two);

  i = __builtin_elementwise_min(one, d);
  // expected-error@-1 {{arguments are of different types ('int' vs 'double')}}

  enum f { three };
  enum f x = __builtin_elementwise_min(one, three);
  // expected-error@-1 {{invalid arithmetic between different enumeration types ('enum e' and 'enum f')}}

  _BitInt(32) ext; // expected-warning {{'_BitInt' in C17 and earlier is a Clang extension}}
  ext = __builtin_elementwise_min(ext, ext);

  const int ci = 0;
  i = __builtin_elementwise_min(ci, i);
  i = __builtin_elementwise_min(i, ci);
  i = __builtin_elementwise_min(ci, ci);

  i = __builtin_elementwise_min(i, int_as_one); // ok (attributes don't match)?
  i = __builtin_elementwise_min(i, b);          // ok (sugar doesn't match)?

  int A[10];
  A = __builtin_elementwise_min(A, A);
  // expected-error@-1 {{1st argument must be a vector, integer or floating-point type (was 'int *')}}

  int(ii);
  int j;
  j = __builtin_elementwise_min(i, j);

  _Complex float c1, c2;
  c1 = __builtin_elementwise_min(c1, c2);
  // expected-error@-1 {{1st argument must be a vector, integer or floating-point type (was '_Complex float')}}
}

void test_builtin_elementwise_maximum(int i, short s, float f, double d, float4 fv, double4 dv, int3 iv, unsigned3 uv, int *p) {
  i = __builtin_elementwise_maximum(p, d);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'int *')}}

  struct Foo foo = __builtin_elementwise_maximum(d, d);
  // expected-error@-1 {{initializing 'struct Foo' with an expression of incompatible type 'double'}}

  i = __builtin_elementwise_maximum(i);
  // expected-error@-1 {{too few arguments to function call, expected 2, have 1}}

  i = __builtin_elementwise_maximum();
  // expected-error@-1 {{too few arguments to function call, expected 2, have 0}}

  i = __builtin_elementwise_maximum(i, i, i);
  // expected-error@-1 {{too many arguments to function call, expected 2, have 3}}

  i = __builtin_elementwise_maximum(fv, iv);
  // expected-error@-1 {{arguments are of different types ('float4' (vector of 4 'float' values) vs 'int3' (vector of 3 'int' values))}}

  i = __builtin_elementwise_maximum(uv, iv);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'unsigned3' (vector of 3 'unsigned int' values))}}

  dv = __builtin_elementwise_maximum(fv, dv);
  // expected-error@-1 {{arguments are of different types ('float4' (vector of 4 'float' values) vs 'double4' (vector of 4 'double' values))}}

  d = __builtin_elementwise_maximum(f, d);
  // expected-error@-1 {{arguments are of different types ('float' vs 'double')}}

  fv = __builtin_elementwise_maximum(fv, fv);

  i = __builtin_elementwise_maximum(iv, iv);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'int3' (vector of 3 'int' values))}}

  i = __builtin_elementwise_maximum(i, i);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'int')}}

  int A[10];
  A = __builtin_elementwise_maximum(A, A);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'int *')}}

  _Complex float c1, c2;
  c1 = __builtin_elementwise_maximum(c1, c2);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was '_Complex float')}}
}

void test_builtin_elementwise_minimum(int i, short s, float f, double d, float4 fv, double4 dv, int3 iv, unsigned3 uv, int *p) {
  i = __builtin_elementwise_minimum(p, d);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'int *')}}

  struct Foo foo = __builtin_elementwise_minimum(d, d);
  // expected-error@-1 {{initializing 'struct Foo' with an expression of incompatible type 'double'}}

  i = __builtin_elementwise_minimum(i);
  // expected-error@-1 {{too few arguments to function call, expected 2, have 1}}

  i = __builtin_elementwise_minimum();
  // expected-error@-1 {{too few arguments to function call, expected 2, have 0}}

  i = __builtin_elementwise_minimum(i, i, i);
  // expected-error@-1 {{too many arguments to function call, expected 2, have 3}}

  i = __builtin_elementwise_minimum(fv, iv);
  // expected-error@-1 {{arguments are of different types ('float4' (vector of 4 'float' values) vs 'int3' (vector of 3 'int' values))}}

  i = __builtin_elementwise_minimum(uv, iv);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'unsigned3' (vector of 3 'unsigned int' values))}}

  dv = __builtin_elementwise_minimum(fv, dv);
  // expected-error@-1 {{arguments are of different types ('float4' (vector of 4 'float' values) vs 'double4' (vector of 4 'double' values))}}

  d = __builtin_elementwise_minimum(f, d);
  // expected-error@-1 {{arguments are of different types ('float' vs 'double')}}

  fv = __builtin_elementwise_minimum(fv, fv);

  i = __builtin_elementwise_minimum(iv, iv);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'int3' (vector of 3 'int' values))}}

  i = __builtin_elementwise_minimum(i, i);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'int')}}

  int A[10];
  A = __builtin_elementwise_minimum(A, A);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'int *')}}

  _Complex float c1, c2;
  c1 = __builtin_elementwise_minimum(c1, c2);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was '_Complex float')}}
}

void test_builtin_elementwise_maximumnum(int i, short s, float f, double d, float4 fv, double4 dv, int3 iv, unsigned3 uv, int *p) {
  i = __builtin_elementwise_maximumnum(p, d);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'int *')}}

  struct Foo foo = __builtin_elementwise_maximumnum(d, d);
  // expected-error@-1 {{initializing 'struct Foo' with an expression of incompatible type 'double'}}

  i = __builtin_elementwise_maximumnum(i);
  // expected-error@-1 {{too few arguments to function call, expected 2, have 1}}

  i = __builtin_elementwise_maximumnum();
  // expected-error@-1 {{too few arguments to function call, expected 2, have 0}}

  i = __builtin_elementwise_maximumnum(i, i, i);
  // expected-error@-1 {{too many arguments to function call, expected 2, have 3}}

  i = __builtin_elementwise_maximumnum(fv, iv);
  // expected-error@-1 {{arguments are of different types ('float4' (vector of 4 'float' values) vs 'int3' (vector of 3 'int' values))}}

  i = __builtin_elementwise_maximumnum(uv, iv);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'unsigned3' (vector of 3 'unsigned int' values))}}

  dv = __builtin_elementwise_maximumnum(fv, dv);
  // expected-error@-1 {{arguments are of different types ('float4' (vector of 4 'float' values) vs 'double4' (vector of 4 'double' values))}}

  d = __builtin_elementwise_maximumnum(f, d);
  // expected-error@-1 {{arguments are of different types ('float' vs 'double')}}

  fv = __builtin_elementwise_maximumnum(fv, fv);

  i = __builtin_elementwise_maximumnum(iv, iv);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'int3' (vector of 3 'int' values))}}

  i = __builtin_elementwise_maximumnum(i, i);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'int')}}

  int A[10];
  A = __builtin_elementwise_maximumnum(A, A);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'int *')}}

  _Complex float c1, c2;
  c1 = __builtin_elementwise_maximumnum(c1, c2);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was '_Complex float')}}
}

void test_builtin_elementwise_minimumnum(int i, short s, float f, double d, float4 fv, double4 dv, int3 iv, unsigned3 uv, int *p) {
  i = __builtin_elementwise_minimumnum(p, d);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'int *')}}

  struct Foo foo = __builtin_elementwise_minimumnum(d, d);
  // expected-error@-1 {{initializing 'struct Foo' with an expression of incompatible type 'double'}}

  i = __builtin_elementwise_minimumnum(i);
  // expected-error@-1 {{too few arguments to function call, expected 2, have 1}}

  i = __builtin_elementwise_minimumnum();
  // expected-error@-1 {{too few arguments to function call, expected 2, have 0}}

  i = __builtin_elementwise_minimumnum(i, i, i);
  // expected-error@-1 {{too many arguments to function call, expected 2, have 3}}

  i = __builtin_elementwise_minimumnum(fv, iv);
  // expected-error@-1 {{arguments are of different types ('float4' (vector of 4 'float' values) vs 'int3' (vector of 3 'int' values))}}

  i = __builtin_elementwise_minimumnum(uv, iv);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'unsigned3' (vector of 3 'unsigned int' values))}}

  dv = __builtin_elementwise_minimumnum(fv, dv);
  // expected-error@-1 {{arguments are of different types ('float4' (vector of 4 'float' values) vs 'double4' (vector of 4 'double' values))}}

  d = __builtin_elementwise_minimumnum(f, d);
  // expected-error@-1 {{arguments are of different types ('float' vs 'double')}}

  fv = __builtin_elementwise_minimumnum(fv, fv);

  i = __builtin_elementwise_minimumnum(iv, iv);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'int3' (vector of 3 'int' values))}}

  i = __builtin_elementwise_minimumnum(i, i);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'int')}}

  int A[10];
  A = __builtin_elementwise_minimumnum(A, A);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'int *')}}

  _Complex float c1, c2;
  c1 = __builtin_elementwise_minimumnum(c1, c2);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was '_Complex float')}}
}

void test_builtin_elementwise_bitreverse(int i, float f, double d, float4 v, int3 iv, unsigned u, unsigned4 uv) {

  struct Foo s = __builtin_elementwise_bitreverse(i);
  // expected-error@-1 {{initializing 'struct Foo' with an expression of incompatible type 'int'}}

  i = __builtin_elementwise_bitreverse();
  // expected-error@-1 {{too few arguments to function call, expected 1, have 0}}

  i = __builtin_elementwise_bitreverse(f);
  // expected-error@-1 {{1st argument must be a scalar or vector of integer types (was 'float')}}
  
  i = __builtin_elementwise_bitreverse(f, f);
  // expected-error@-1 {{too many arguments to function call, expected 1, have 2}}

  u = __builtin_elementwise_bitreverse(d);
  // expected-error@-1 {{1st argument must be a scalar or vector of integer types (was 'double')}}

  v = __builtin_elementwise_bitreverse(v);
  // expected-error@-1 {{1st argument must be a scalar or vector of integer types (was 'float4' (vector of 4 'float' values))}}
}

void test_builtin_elementwise_ceil(int i, float f, double d, float4 v, int3 iv, unsigned u, unsigned4 uv) {

  struct Foo s = __builtin_elementwise_ceil(f);
  // expected-error@-1 {{initializing 'struct Foo' with an expression of incompatible type 'float'}}

  i = __builtin_elementwise_ceil();
  // expected-error@-1 {{too few arguments to function call, expected 1, have 0}}

  i = __builtin_elementwise_ceil(i);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'int')}}

  i = __builtin_elementwise_ceil(f, f);
  // expected-error@-1 {{too many arguments to function call, expected 1, have 2}}

  u = __builtin_elementwise_ceil(u);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'unsigned int')}}

  uv = __builtin_elementwise_ceil(uv);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'unsigned4' (vector of 4 'unsigned int' values))}}
}

void test_builtin_elementwise_acos(int i, float f, double d, float4 v, int3 iv, unsigned u, unsigned4 uv) {

  struct Foo s = __builtin_elementwise_acos(f);
  // expected-error@-1 {{initializing 'struct Foo' with an expression of incompatible type 'float'}}

  i = __builtin_elementwise_acos();
  // expected-error@-1 {{too few arguments to function call, expected 1, have 0}}

  i = __builtin_elementwise_acos(i);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'int')}}

  i = __builtin_elementwise_acos(f, f);
  // expected-error@-1 {{too many arguments to function call, expected 1, have 2}}

  u = __builtin_elementwise_acos(u);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'unsigned int')}}

  uv = __builtin_elementwise_acos(uv);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'unsigned4' (vector of 4 'unsigned int' values))}}
}

void test_builtin_elementwise_cos(int i, float f, double d, float4 v, int3 iv, unsigned u, unsigned4 uv) {

  struct Foo s = __builtin_elementwise_cos(f);
  // expected-error@-1 {{initializing 'struct Foo' with an expression of incompatible type 'float'}}

  i = __builtin_elementwise_cos();
  // expected-error@-1 {{too few arguments to function call, expected 1, have 0}}

  i = __builtin_elementwise_cos(i);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'int')}}

  i = __builtin_elementwise_cos(f, f);
  // expected-error@-1 {{too many arguments to function call, expected 1, have 2}}

  u = __builtin_elementwise_cos(u);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'unsigned int')}}

  uv = __builtin_elementwise_cos(uv);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'unsigned4' (vector of 4 'unsigned int' values))}}
}

void test_builtin_elementwise_cosh(int i, float f, double d, float4 v, int3 iv, unsigned u, unsigned4 uv) {

  struct Foo s = __builtin_elementwise_cosh(f);
  // expected-error@-1 {{initializing 'struct Foo' with an expression of incompatible type 'float'}}

  i = __builtin_elementwise_cosh();
  // expected-error@-1 {{too few arguments to function call, expected 1, have 0}}

  i = __builtin_elementwise_cosh(i);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'int')}}

  i = __builtin_elementwise_cosh(f, f);
  // expected-error@-1 {{too many arguments to function call, expected 1, have 2}}

  u = __builtin_elementwise_cosh(u);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'unsigned int')}}

  uv = __builtin_elementwise_cosh(uv);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'unsigned4' (vector of 4 'unsigned int' values))}}
}

void test_builtin_elementwise_exp(int i, float f, double d, float4 v, int3 iv, unsigned u, unsigned4 uv) {

  struct Foo s = __builtin_elementwise_exp(f);
  // expected-error@-1 {{initializing 'struct Foo' with an expression of incompatible type 'float'}}

  i = __builtin_elementwise_exp();
  // expected-error@-1 {{too few arguments to function call, expected 1, have 0}}

  i = __builtin_elementwise_exp(i);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'int')}}

  i = __builtin_elementwise_exp(f, f);
  // expected-error@-1 {{too many arguments to function call, expected 1, have 2}}

  u = __builtin_elementwise_exp(u);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'unsigned int')}}

  uv = __builtin_elementwise_exp(uv);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'unsigned4' (vector of 4 'unsigned int' values))}}
}

void test_builtin_elementwise_exp2(int i, float f, double d, float4 v, int3 iv, unsigned u, unsigned4 uv) {

  struct Foo s = __builtin_elementwise_exp2(f);
  // expected-error@-1 {{initializing 'struct Foo' with an expression of incompatible type 'float'}}

  i = __builtin_elementwise_exp2();
  // expected-error@-1 {{too few arguments to function call, expected 1, have 0}}

  i = __builtin_elementwise_exp2(i);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'int')}}

  i = __builtin_elementwise_exp2(f, f);
  // expected-error@-1 {{too many arguments to function call, expected 1, have 2}}

  u = __builtin_elementwise_exp2(u);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'unsigned int')}}

  uv = __builtin_elementwise_exp2(uv);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'unsigned4' (vector of 4 'unsigned int' values))}}
}

void test_builtin_elementwise_exp10(int i, float f, double d, float4 v, int3 iv, unsigned u, unsigned4 uv) {

  struct Foo s = __builtin_elementwise_exp10(f);
  // expected-error@-1 {{initializing 'struct Foo' with an expression of incompatible type 'float'}}

  i = __builtin_elementwise_exp10();
  // expected-error@-1 {{too few arguments to function call, expected 1, have 0}}

  i = __builtin_elementwise_exp10(i);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'int')}}

  i = __builtin_elementwise_exp10(f, f);
  // expected-error@-1 {{too many arguments to function call, expected 1, have 2}}

  u = __builtin_elementwise_exp10(u);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'unsigned int')}}

  uv = __builtin_elementwise_exp10(uv);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'unsigned4' (vector of 4 'unsigned int' values))}}
}

void test_builtin_elementwise_floor(int i, float f, double d, float4 v, int3 iv, unsigned u, unsigned4 uv) {

  struct Foo s = __builtin_elementwise_floor(f);
  // expected-error@-1 {{initializing 'struct Foo' with an expression of incompatible type 'float'}}

  i = __builtin_elementwise_floor();
  // expected-error@-1 {{too few arguments to function call, expected 1, have 0}}

  i = __builtin_elementwise_floor(i);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'int')}}

  i = __builtin_elementwise_floor(f, f);
  // expected-error@-1 {{too many arguments to function call, expected 1, have 2}}

  u = __builtin_elementwise_floor(u);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'unsigned int')}}

  uv = __builtin_elementwise_floor(uv);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'unsigned4' (vector of 4 'unsigned int' values))}}
}

void test_builtin_elementwise_log(int i, float f, double d, float4 v, int3 iv, unsigned u, unsigned4 uv) {

  struct Foo s = __builtin_elementwise_log(f);
  // expected-error@-1 {{initializing 'struct Foo' with an expression of incompatible type 'float'}}

  i = __builtin_elementwise_log();
  // expected-error@-1 {{too few arguments to function call, expected 1, have 0}}

  i = __builtin_elementwise_log(i);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'int')}}

  i = __builtin_elementwise_log(f, f);
  // expected-error@-1 {{too many arguments to function call, expected 1, have 2}}

  u = __builtin_elementwise_log(u);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'unsigned int')}}

  uv = __builtin_elementwise_log(uv);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'unsigned4' (vector of 4 'unsigned int' values))}}
}

void test_builtin_elementwise_log10(int i, float f, double d, float4 v, int3 iv, unsigned u, unsigned4 uv) {

  struct Foo s = __builtin_elementwise_log10(f);
  // expected-error@-1 {{initializing 'struct Foo' with an expression of incompatible type 'float'}}

  i = __builtin_elementwise_log10();
  // expected-error@-1 {{too few arguments to function call, expected 1, have 0}}

  i = __builtin_elementwise_log10(i);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'int')}}

  i = __builtin_elementwise_log10(f, f);
  // expected-error@-1 {{too many arguments to function call, expected 1, have 2}}

  u = __builtin_elementwise_log10(u);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'unsigned int')}}

  uv = __builtin_elementwise_log10(uv);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'unsigned4' (vector of 4 'unsigned int' values))}}
}

void test_builtin_elementwise_log2(int i, float f, double d, float4 v, int3 iv, unsigned u, unsigned4 uv) {

  struct Foo s = __builtin_elementwise_log2(f);
  // expected-error@-1 {{initializing 'struct Foo' with an expression of incompatible type 'float'}}

  i = __builtin_elementwise_log2();
  // expected-error@-1 {{too few arguments to function call, expected 1, have 0}}

  i = __builtin_elementwise_log2(i);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'int')}}

  i = __builtin_elementwise_log2(f, f);
  // expected-error@-1 {{too many arguments to function call, expected 1, have 2}}

  u = __builtin_elementwise_log2(u);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'unsigned int')}}

  uv = __builtin_elementwise_log2(uv);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'unsigned4' (vector of 4 'unsigned int' values))}}
}

void test_builtin_elementwise_popcount(int i, float f, double d, float4 v, int3 iv, unsigned u, unsigned4 uv) {

  struct Foo s = __builtin_elementwise_popcount(i);
  // expected-error@-1 {{initializing 'struct Foo' with an expression of incompatible type 'int'}}

  i = __builtin_elementwise_popcount();
  // expected-error@-1 {{too few arguments to function call, expected 1, have 0}}

  i = __builtin_elementwise_popcount(f);
  // expected-error@-1 {{1st argument must be a scalar or vector of integer types (was 'float')}}

  i = __builtin_elementwise_popcount(f, f);
  // expected-error@-1 {{too many arguments to function call, expected 1, have 2}}

  u = __builtin_elementwise_popcount(d);
  // expected-error@-1 {{1st argument must be a scalar or vector of integer types (was 'double')}}

  v = __builtin_elementwise_popcount(v);
  // expected-error@-1 {{1st argument must be a scalar or vector of integer types (was 'float4' (vector of 4 'float' values))}}

  int2 i2 = __builtin_elementwise_popcount(iv);
  // expected-error@-1 {{initializing 'int2' (vector of 2 'int' values) with an expression of incompatible type 'int3' (vector of 3 'int' values)}}

  iv = __builtin_elementwise_popcount(i2);
  // expected-error@-1 {{assigning to 'int3' (vector of 3 'int' values) from incompatible type 'int2' (vector of 2 'int' values)}}

  unsigned3 u3 = __builtin_elementwise_popcount(iv);
  // expected-error@-1 {{initializing 'unsigned3' (vector of 3 'unsigned int' values) with an expression of incompatible type 'int3' (vector of 3 'int' values)}}

  iv = __builtin_elementwise_popcount(u3);
  // expected-error@-1 {{assigning to 'int3' (vector of 3 'int' values) from incompatible type 'unsigned3' (vector of 3 'unsigned int' values)}}
}

void test_builtin_elementwise_fmod(int i, short s, double d, float4 v, int3 iv, unsigned3 uv, int *p) {
  i = __builtin_elementwise_fmod(p, d);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'int *')}}

  struct Foo foo = __builtin_elementwise_fmod(i, i);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'int')}}

  i = __builtin_elementwise_fmod(i);
  // expected-error@-1 {{too few arguments to function call, expected 2, have 1}}

  i = __builtin_elementwise_fmod();
  // expected-error@-1 {{too few arguments to function call, expected 2, have 0}}

  i = __builtin_elementwise_fmod(i, i, i);
  // expected-error@-1 {{too many arguments to function call, expected 2, have 3}}

  i = __builtin_elementwise_fmod(v, iv);
  // expected-error@-1 {{arguments are of different types ('float4' (vector of 4 'float' values) vs 'int3' (vector of 3 'int' values))}}

  i = __builtin_elementwise_fmod(uv, iv);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'unsigned3' (vector of 3 'unsigned int' values))}}

  i = __builtin_elementwise_fmod(d, v);
  // expected-error@-1 {{arguments are of different types ('double' vs 'float4' (vector of 4 'float' values))}}
}

void test_builtin_elementwise_pow(int i, short s, double d, float4 v, int3 iv, unsigned3 uv, int *p) {
  i = __builtin_elementwise_pow(p, d);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'int *')}}

  struct Foo foo = __builtin_elementwise_pow(i, i);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'int')}}

  i = __builtin_elementwise_pow(i);
  // expected-error@-1 {{too few arguments to function call, expected 2, have 1}}

  i = __builtin_elementwise_pow();
  // expected-error@-1 {{too few arguments to function call, expected 2, have 0}}

  i = __builtin_elementwise_pow(i, i, i);
  // expected-error@-1 {{too many arguments to function call, expected 2, have 3}}

  i = __builtin_elementwise_pow(v, iv);
  // expected-error@-1 {{arguments are of different types ('float4' (vector of 4 'float' values) vs 'int3' (vector of 3 'int' values))}}

  i = __builtin_elementwise_pow(uv, iv);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'unsigned3' (vector of 3 'unsigned int' values))}}
  
}

void test_builtin_elementwise_roundeven(int i, float f, double d, float4 v, int3 iv, unsigned u, unsigned4 uv) {

  struct Foo s = __builtin_elementwise_roundeven(f);
  // expected-error@-1 {{initializing 'struct Foo' with an expression of incompatible type 'float'}}

  i = __builtin_elementwise_roundeven();
  // expected-error@-1 {{too few arguments to function call, expected 1, have 0}}

  i = __builtin_elementwise_roundeven(i);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'int')}}

  i = __builtin_elementwise_roundeven(f, f);
  // expected-error@-1 {{too many arguments to function call, expected 1, have 2}}

  u = __builtin_elementwise_roundeven(u);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'unsigned int')}}

  uv = __builtin_elementwise_roundeven(uv);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'unsigned4' (vector of 4 'unsigned int' values))}}
}

void test_builtin_elementwise_round(int i, float f, double d, float4 v, int3 iv, unsigned u, unsigned4 uv) {
  struct Foo s = __builtin_elementwise_round(f);
  // expected-error@-1 {{initializing 'struct Foo' with an expression of incompatible type 'float'}}

  i = __builtin_elementwise_round();
  // expected-error@-1 {{too few arguments to function call, expected 1, have 0}}

  i = __builtin_elementwise_round(i);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'int')}}

  i = __builtin_elementwise_round(f, f);
  // expected-error@-1 {{too many arguments to function call, expected 1, have 2}}

  u = __builtin_elementwise_round(u);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'unsigned int')}}

  uv = __builtin_elementwise_round(uv);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'unsigned4' (vector of 4 'unsigned int' values))}}

  _Complex float c1, c2;
  c1 = __builtin_elementwise_round(c1);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was '_Complex float')}}
}

void test_builtin_elementwise_rint(int i, float f, double d, float4 v, int3 iv, unsigned u, unsigned4 uv) {
  struct Foo s = __builtin_elementwise_rint(f);
  // expected-error@-1 {{initializing 'struct Foo' with an expression of incompatible type 'float'}}

  i = __builtin_elementwise_rint();
  // expected-error@-1 {{too few arguments to function call, expected 1, have 0}}

  i = __builtin_elementwise_rint(i);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'int')}}

  i = __builtin_elementwise_rint(f, f);
  // expected-error@-1 {{too many arguments to function call, expected 1, have 2}}

  u = __builtin_elementwise_rint(u);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'unsigned int')}}

  uv = __builtin_elementwise_rint(uv);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'unsigned4' (vector of 4 'unsigned int' values))}}

  _Complex float c1, c2;
  c1 = __builtin_elementwise_rint(c1);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was '_Complex float')}}
}

void test_builtin_elementwise_nearbyint(int i, float f, double d, float4 v, int3 iv, unsigned u, unsigned4 uv) {
  struct Foo s = __builtin_elementwise_nearbyint(f);
  // expected-error@-1 {{initializing 'struct Foo' with an expression of incompatible type 'float'}}

  i = __builtin_elementwise_nearbyint();
  // expected-error@-1 {{too few arguments to function call, expected 1, have 0}}

  i = __builtin_elementwise_nearbyint(i);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'int')}}

  i = __builtin_elementwise_nearbyint(f, f);
  // expected-error@-1 {{too many arguments to function call, expected 1, have 2}}

  u = __builtin_elementwise_nearbyint(u);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'unsigned int')}}

  uv = __builtin_elementwise_nearbyint(uv);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'unsigned4' (vector of 4 'unsigned int' values))}}

  _Complex float c1, c2;
  c1 = __builtin_elementwise_nearbyint(c1);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was '_Complex float')}}
}

void test_builtin_elementwise_asin(int i, float f, double d, float4 v, int3 iv, unsigned u, unsigned4 uv) {

  struct Foo s = __builtin_elementwise_asin(f);
  // expected-error@-1 {{initializing 'struct Foo' with an expression of incompatible type 'float'}}

  i = __builtin_elementwise_asin();
  // expected-error@-1 {{too few arguments to function call, expected 1, have 0}}

  i = __builtin_elementwise_asin(i);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'int')}}

  i = __builtin_elementwise_asin(f, f);
  // expected-error@-1 {{too many arguments to function call, expected 1, have 2}}

  u = __builtin_elementwise_asin(u);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'unsigned int')}}

  uv = __builtin_elementwise_asin(uv);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'unsigned4' (vector of 4 'unsigned int' values))}}
}

void test_builtin_elementwise_sin(int i, float f, double d, float4 v, int3 iv, unsigned u, unsigned4 uv) {

  struct Foo s = __builtin_elementwise_sin(f);
  // expected-error@-1 {{initializing 'struct Foo' with an expression of incompatible type 'float'}}

  i = __builtin_elementwise_sin();
  // expected-error@-1 {{too few arguments to function call, expected 1, have 0}}

  i = __builtin_elementwise_sin(i);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'int')}}

  i = __builtin_elementwise_sin(f, f);
  // expected-error@-1 {{too many arguments to function call, expected 1, have 2}}

  u = __builtin_elementwise_sin(u);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'unsigned int')}}

  uv = __builtin_elementwise_sin(uv);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'unsigned4' (vector of 4 'unsigned int' values))}}
}

void test_builtin_elementwise_sinh(int i, float f, double d, float4 v, int3 iv, unsigned u, unsigned4 uv) {

  struct Foo s = __builtin_elementwise_sinh(f);
  // expected-error@-1 {{initializing 'struct Foo' with an expression of incompatible type 'float'}}

  i = __builtin_elementwise_sinh();
  // expected-error@-1 {{too few arguments to function call, expected 1, have 0}}

  i = __builtin_elementwise_sinh(i);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'int')}}

  i = __builtin_elementwise_sinh(f, f);
  // expected-error@-1 {{too many arguments to function call, expected 1, have 2}}

  u = __builtin_elementwise_sinh(u);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'unsigned int')}}

  uv = __builtin_elementwise_sinh(uv);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'unsigned4' (vector of 4 'unsigned int' values))}}
}

void test_builtin_elementwise_sqrt(int i, float f, double d, float4 v, int3 iv, unsigned u, unsigned4 uv) {

  struct Foo s = __builtin_elementwise_sqrt(f);
  // expected-error@-1 {{initializing 'struct Foo' with an expression of incompatible type 'float'}}

  i = __builtin_elementwise_sqrt();
  // expected-error@-1 {{too few arguments to function call, expected 1, have 0}}

  i = __builtin_elementwise_sqrt(i);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'int')}}

  i = __builtin_elementwise_sqrt(f, f);
  // expected-error@-1 {{too many arguments to function call, expected 1, have 2}}

  u = __builtin_elementwise_sqrt(u);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'unsigned int')}}

  uv = __builtin_elementwise_sqrt(uv);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'unsigned4' (vector of 4 'unsigned int' values))}}
}

void test_builtin_elementwise_atan(int i, float f, double d, float4 v, int3 iv, unsigned u, unsigned4 uv) {

  struct Foo s = __builtin_elementwise_atan(f);
  // expected-error@-1 {{initializing 'struct Foo' with an expression of incompatible type 'float'}}

  i = __builtin_elementwise_atan();
  // expected-error@-1 {{too few arguments to function call, expected 1, have 0}}

  i = __builtin_elementwise_atan(i);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'int')}}

  i = __builtin_elementwise_atan(f, f);
  // expected-error@-1 {{too many arguments to function call, expected 1, have 2}}

  u = __builtin_elementwise_atan(u);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'unsigned int')}}

  uv = __builtin_elementwise_atan(uv);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'unsigned4' (vector of 4 'unsigned int' values))}}
}

void test_builtin_elementwise_atan2(int i, float f, double d, float4 v, int3 iv, unsigned u, unsigned4 uv) {

  struct Foo s = __builtin_elementwise_atan2(f, f);
  // expected-error@-1 {{initializing 'struct Foo' with an expression of incompatible type 'float'}}

  i = __builtin_elementwise_atan2();
  // expected-error@-1 {{too few arguments to function call, expected 2, have 0}}

  i = __builtin_elementwise_atan2(f);
  // expected-error@-1 {{too few arguments to function call, expected 2, have 1}}

  i = __builtin_elementwise_atan2(i, i);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'int')}}

  i = __builtin_elementwise_atan2(f, f, f);
  // expected-error@-1 {{too many arguments to function call, expected 2, have 3}}

  u = __builtin_elementwise_atan2(u, u);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'unsigned int')}}

  uv = __builtin_elementwise_atan2(uv, uv);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'unsigned4' (vector of 4 'unsigned int' values))}}
}

void test_builtin_elementwise_tan(int i, float f, double d, float4 v, int3 iv, unsigned u, unsigned4 uv) {

  struct Foo s = __builtin_elementwise_tan(f);
  // expected-error@-1 {{initializing 'struct Foo' with an expression of incompatible type 'float'}}

  i = __builtin_elementwise_tan();
  // expected-error@-1 {{too few arguments to function call, expected 1, have 0}}

  i = __builtin_elementwise_tan(i);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'int')}}

  i = __builtin_elementwise_tan(f, f);
  // expected-error@-1 {{too many arguments to function call, expected 1, have 2}}

  u = __builtin_elementwise_tan(u);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'unsigned int')}}

  uv = __builtin_elementwise_tan(uv);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'unsigned4' (vector of 4 'unsigned int' values))}}
}

void test_builtin_elementwise_tanh(int i, float f, double d, float4 v, int3 iv, unsigned u, unsigned4 uv) {

  struct Foo s = __builtin_elementwise_tanh(f);
  // expected-error@-1 {{initializing 'struct Foo' with an expression of incompatible type 'float'}}

  i = __builtin_elementwise_tanh();
  // expected-error@-1 {{too few arguments to function call, expected 1, have 0}}

  i = __builtin_elementwise_tanh(i);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'int')}}

  i = __builtin_elementwise_tanh(f, f);
  // expected-error@-1 {{too many arguments to function call, expected 1, have 2}}

  u = __builtin_elementwise_tanh(u);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'unsigned int')}}

  uv = __builtin_elementwise_tanh(uv);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'unsigned4' (vector of 4 'unsigned int' values))}}
}

void test_builtin_elementwise_trunc(int i, float f, double d, float4 v, int3 iv, unsigned u, unsigned4 uv) {

  struct Foo s = __builtin_elementwise_trunc(f);
  // expected-error@-1 {{initializing 'struct Foo' with an expression of incompatible type 'float'}}

  i = __builtin_elementwise_trunc();
  // expected-error@-1 {{too few arguments to function call, expected 1, have 0}}

  i = __builtin_elementwise_trunc(i);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'int')}}

  i = __builtin_elementwise_trunc(f, f);
  // expected-error@-1 {{too many arguments to function call, expected 1, have 2}}

  u = __builtin_elementwise_trunc(u);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'unsigned int')}}

  uv = __builtin_elementwise_trunc(uv);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'unsigned4' (vector of 4 'unsigned int' values))}}
}

void test_builtin_elementwise_canonicalize(int i, float f, double d, float4 v, int3 iv, unsigned u, unsigned4 uv) {

  struct Foo s = __builtin_elementwise_canonicalize(f);
  // expected-error@-1 {{initializing 'struct Foo' with an expression of incompatible type 'float'}}

  i = __builtin_elementwise_canonicalize();
  // expected-error@-1 {{too few arguments to function call, expected 1, have 0}}

  i = __builtin_elementwise_canonicalize(i);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'int')}}

  i = __builtin_elementwise_canonicalize(f, f);
  // expected-error@-1 {{too many arguments to function call, expected 1, have 2}}

  u = __builtin_elementwise_canonicalize(u);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'unsigned int')}}

  uv = __builtin_elementwise_canonicalize(uv);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'unsigned4' (vector of 4 'unsigned int' values))}}
}

void test_builtin_elementwise_copysign(int i, short s, double d, float f, float4 v, int3 iv, unsigned3 uv, int *p) {
  i = __builtin_elementwise_copysign(p, d);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'int *')}}

  i = __builtin_elementwise_copysign(i, i);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'int')}}

  i = __builtin_elementwise_copysign(i);
  // expected-error@-1 {{too few arguments to function call, expected 2, have 1}}

  i = __builtin_elementwise_copysign();
  // expected-error@-1 {{too few arguments to function call, expected 2, have 0}}

  i = __builtin_elementwise_copysign(i, i, i);
  // expected-error@-1 {{too many arguments to function call, expected 2, have 3}}

  i = __builtin_elementwise_copysign(v, iv);
  // expected-error@-1 {{2nd argument must be a scalar or vector of floating-point types (was 'int3' (vector of 3 'int' values))}}

  i = __builtin_elementwise_copysign(uv, iv);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'unsigned3' (vector of 3 'unsigned int' values))}}

  s = __builtin_elementwise_copysign(i, s);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'int')}}

  f = __builtin_elementwise_copysign(f, i);
  // expected-error@-1 {{2nd argument must be a scalar or vector of floating-point types (was 'int')}}

  f = __builtin_elementwise_copysign(i, f);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'int')}}

  enum e { one,
           two };
  i = __builtin_elementwise_copysign(one, two);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'int')}}

  enum f { three };
  enum f x = __builtin_elementwise_copysign(one, three);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'int')}}

  _BitInt(32) ext; // expected-warning {{'_BitInt' in C17 and earlier is a Clang extension}}
  ext = __builtin_elementwise_copysign(ext, ext);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was '_BitInt(32)')}}

  const float cf32 = 0.0f;
  f = __builtin_elementwise_copysign(cf32, f);
  f = __builtin_elementwise_copysign(f, cf32);
  f = __builtin_elementwise_copysign(cf32, f);

  f = __builtin_elementwise_copysign(f, float_as_one); // ok (attributes don't match)?
  f = __builtin_elementwise_copysign(f, waf);          // ok (sugar doesn't match)?

  float A[10];
  A = __builtin_elementwise_copysign(A, A);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'float *')}}

  float(ii);
  float j;
  j = __builtin_elementwise_copysign(f, j);

  _Complex float c1, c2;
  c1 = __builtin_elementwise_copysign(c1, c2);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was '_Complex float')}}

  double f64 = 0.0;
  double tmp0 = __builtin_elementwise_copysign(f64, f);
  // expected-error@-1 {{arguments are of different types ('double' vs 'float')}}

  float tmp1 = __builtin_elementwise_copysign(f, f64);
  //expected-error@-1 {{arguments are of different types ('float' vs 'double')}}

  float4 v4f32 = 0.0f;
  float4 tmp2 = __builtin_elementwise_copysign(v4f32, f);
  // expected-error@-1 {{arguments are of different types ('float4' (vector of 4 'float' values) vs 'float')}}

  float tmp3 = __builtin_elementwise_copysign(f, v4f32);
  // expected-error@-1 {{arguments are of different types ('float' vs 'float4' (vector of 4 'float' values))}}

  float2 v2f32 = 0.0f;
  double4 v4f64 = 0.0;
  double4 tmp4 = __builtin_elementwise_copysign(v4f64, v4f32);
  // expected-error@-1 {{arguments are of different types ('double4' (vector of 4 'double' values) vs 'float4' (vector of 4 'float' values))}}

  float4 tmp6 = __builtin_elementwise_copysign(v4f32, v4f64);
  // expected-error@-1 {{arguments are of different types ('float4' (vector of 4 'float' values) vs 'double4' (vector of 4 'double' values))}}

  float4 tmp7 = __builtin_elementwise_copysign(v4f32, v2f32);
  // expected-error@-1 {{arguments are of different types ('float4' (vector of 4 'float' values) vs 'float2' (vector of 2 'float' values))}}

  float2 tmp8 = __builtin_elementwise_copysign(v2f32, v4f32);
  // expected-error@-1 {{arguments are of different types ('float2' (vector of 2 'float' values) vs 'float4' (vector of 4 'float' values))}}

  float2 tmp9 = __builtin_elementwise_copysign(v4f32, v4f32);
  // expected-error@-1 {{initializing 'float2' (vector of 2 'float' values) with an expression of incompatible type 'float4' (vector of 4 'float' values)}}
}

void test_builtin_elementwise_fma(int i32, int2 v2i32, short i16,
                                  double f64, double2 v2f64, double2 v3f64,
                                  float f32, float2 v2f32, float v3f32, float4 v4f32,
                                  const float4 c_v4f32,
                                  int3 v3i32, int *ptr) {

  f32 = __builtin_elementwise_fma();
  // expected-error@-1 {{too few arguments to function call, expected 3, have 0}}

  f32 = __builtin_elementwise_fma(f32);
  // expected-error@-1 {{too few arguments to function call, expected 3, have 1}}

  f32 = __builtin_elementwise_fma(f32, f32);
  // expected-error@-1 {{too few arguments to function call, expected 3, have 2}}

  f32 = __builtin_elementwise_fma(f32, f32, f32, f32);
  // expected-error@-1 {{too many arguments to function call, expected 3, have 4}}

  f32 = __builtin_elementwise_fma(f64, f32, f32);
  // expected-error@-1 {{arguments are of different types ('double' vs 'float')}}

  f32 = __builtin_elementwise_fma(f32, f64, f32);
  // expected-error@-1 {{arguments are of different types ('float' vs 'double')}}

  f32 = __builtin_elementwise_fma(f32, f32, f64);
  // expected-error@-1 {{arguments are of different types ('float' vs 'double')}}

  f32 = __builtin_elementwise_fma(f32, f32, f64);
  // expected-error@-1 {{arguments are of different types ('float' vs 'double')}}

  f64 = __builtin_elementwise_fma(f64, f32, f32);
  // expected-error@-1 {{arguments are of different types ('double' vs 'float')}}

  f64 = __builtin_elementwise_fma(f64, f64, f32);
  // expected-error@-1 {{arguments are of different types ('double' vs 'float')}}

  f64 = __builtin_elementwise_fma(f64, f32, f64);
  // expected-error@-1 {{arguments are of different types ('double' vs 'float')}}

  v2f64 = __builtin_elementwise_fma(v2f32, f64, f64);
  // expected-error@-1 {{arguments are of different types ('float2' (vector of 2 'float' values) vs 'double'}}

  v2f64 = __builtin_elementwise_fma(v2f32, v2f64, f64);
  // expected-error@-1 {{arguments are of different types ('float2' (vector of 2 'float' values) vs 'double2' (vector of 2 'double' values)}}

  v2f64 = __builtin_elementwise_fma(v2f32, f64, v2f64);
  // expected-error@-1 {{arguments are of different types ('float2' (vector of 2 'float' values) vs 'double'}}

  v2f64 = __builtin_elementwise_fma(f64, v2f32, v2f64);
  // expected-error@-1 {{arguments are of different types ('double' vs 'float2' (vector of 2 'float' values)}}

  v2f64 = __builtin_elementwise_fma(f64, v2f64, v2f64);
  // expected-error@-1 {{arguments are of different types ('double' vs 'double2' (vector of 2 'double' values)}}

  i32 = __builtin_elementwise_fma(i32, i32, i32);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'int')}}

  v2i32 = __builtin_elementwise_fma(v2i32, v2i32, v2i32);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'int2' (vector of 2 'int' values))}}

  f32 = __builtin_elementwise_fma(f32, f32, i32);
  // expected-error@-1 {{3rd argument must be a scalar or vector of floating-point types (was 'int')}}

  f32 = __builtin_elementwise_fma(f32, i32, f32);
  // expected-error@-1 {{2nd argument must be a scalar or vector of floating-point types (was 'int')}}

  f32 = __builtin_elementwise_fma(f32, f32, i32);
  // expected-error@-1 {{3rd argument must be a scalar or vector of floating-point types (was 'int')}}


  _Complex float c1, c2, c3;
  c1 = __builtin_elementwise_fma(c1, f32, f32);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was '_Complex float')}}

  c2 = __builtin_elementwise_fma(f32, c2, f32);
  // expected-error@-1 {{2nd argument must be a scalar or vector of floating-point types (was '_Complex float')}}

  c3 = __builtin_elementwise_fma(f32, f32, c3);
  // expected-error@-1 {{3rd argument must be a scalar or vector of floating-point types (was '_Complex float')}}
}

void test_builtin_elementwise_fsh(int i32, int2 v2i32, short i16, int3 v3i32,
				  double f64, float f32, float2 v2f32) {
    i32 = __builtin_elementwise_fshl();
    // expected-error@-1 {{too few arguments to function call, expected 3, have 0}}

    i32 = __builtin_elementwise_fshr();
    // expected-error@-1 {{too few arguments to function call, expected 3, have 0}}

    i32 = __builtin_elementwise_fshl(i32, i32);
    // expected-error@-1 {{too few arguments to function call, expected 3, have 2}}

    i32 = __builtin_elementwise_fshr(i32, i32);
    // expected-error@-1 {{too few arguments to function call, expected 3, have 2}}

    i32 = __builtin_elementwise_fshl(i32, i32, i16);
    // expected-error@-1 {{arguments are of different types ('int' vs 'short')}}

    i16 = __builtin_elementwise_fshr(i16, i32, i16);
    // expected-error@-1 {{arguments are of different types ('short' vs 'int')}}

    f32 = __builtin_elementwise_fshl(f32, f32, f32);
    // expected-error@-1 {{argument must be a scalar or vector of integer types (was 'float')}}

    f64 = __builtin_elementwise_fshr(f64, f64, f64);
    // expected-error@-1 {{argument must be a scalar or vector of integer types (was 'double')}}

    v2i32 = __builtin_elementwise_fshl(v2i32, v2i32, v2f32);
    // expected-error@-1 {{argument must be a scalar or vector of integer types (was 'float2' (vector of 2 'float' values))}}

    v2i32 = __builtin_elementwise_fshr(v2i32, v2i32, v3i32);
    // expected-error@-1 {{arguments are of different types ('int2' (vector of 2 'int' values) vs 'int3' (vector of 3 'int' values))}}

    v3i32 = __builtin_elementwise_fshl(v3i32, v3i32, v2i32);
    // expected-error@-1 {{arguments are of different types ('int3' (vector of 3 'int' values) vs 'int2' (vector of 2 'int' values))}}
}

typedef struct {
  float3 b;
} struct_float3;
// This example uncovered a bug #141397 :
// Type mismatch error when 'builtin-elementwise-math' arguments have different qualifiers, this should be well-formed
float3 foo(float3 a,const struct_float3* hi) {
  float3 b = __builtin_elementwise_max((float3)(0.0f), a);
  return __builtin_elementwise_pow(b, hi->b.yyy);
}

void test_builtin_elementwise_ctlz(int i32, int2 v2i32, short i16,
                                   double f64, double2 v2f64) {
  f64 = __builtin_elementwise_ctlz(f64);
  // expected-error@-1 {{1st argument must be a scalar or vector of integer types (was 'double')}}

  _Complex float c1;
  c1 = __builtin_elementwise_ctlz(c1);
  // expected-error@-1 {{1st argument must be a scalar or vector of integer types (was '_Complex float')}}

  v2i32 = __builtin_elementwise_ctlz(v2i32, i32);
  // expected-error@-1 {{arguments are of different types ('int2' (vector of 2 'int' values) vs 'int')}}

  v2i32 = __builtin_elementwise_ctlz(v2i32, f64);
  // expected-error@-1 {{arguments are of different types ('int2' (vector of 2 'int' values) vs 'double')}}

  v2i32 = __builtin_elementwise_ctlz();
  // expected-error@-1 {{too few arguments to function call, expected 1, have 0}}

  v2i32 = __builtin_elementwise_ctlz(v2i32, v2i32, f64);
  // expected-error@-1 {{too many arguments to function call, expected 2, have 3}}
}

void test_builtin_elementwise_cttz(int i32, int2 v2i32, short i16,
                                   double f64, double2 v2f64) {
  f64 = __builtin_elementwise_cttz(f64);
  // expected-error@-1 {{1st argument must be a scalar or vector of integer types (was 'double')}}

  _Complex float c1;
  c1 = __builtin_elementwise_cttz(c1);
  // expected-error@-1 {{1st argument must be a scalar or vector of integer types (was '_Complex float')}}

  v2i32 = __builtin_elementwise_cttz(v2i32, i32);
  // expected-error@-1 {{arguments are of different types ('int2' (vector of 2 'int' values) vs 'int')}}

  v2i32 = __builtin_elementwise_cttz(v2i32, f64);
  // expected-error@-1 {{arguments are of different types ('int2' (vector of 2 'int' values) vs 'double')}}

  v2i32 = __builtin_elementwise_cttz();
  // expected-error@-1 {{too few arguments to function call, expected 1, have 0}}

  v2i32 = __builtin_elementwise_cttz(v2i32, v2i32, f64);
  // expected-error@-1 {{too many arguments to function call, expected 2, have 3}}
}
