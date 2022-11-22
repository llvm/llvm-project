// RUN: %clang_cc1 -std=c99 %s -pedantic -verify -triple=x86_64-apple-darwin9

typedef float float4 __attribute__((ext_vector_type(4)));
typedef int int3 __attribute__((ext_vector_type(3)));
typedef unsigned unsigned3 __attribute__((ext_vector_type(3)));
typedef unsigned unsigned4 __attribute__((ext_vector_type(4)));

struct Foo {
  char *p;
};

__attribute__((address_space(1))) int int_as_one;
typedef int bar;
bar b;

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
  // expected-error@-1 {{1st argument must be a signed integer or floating point type (was 'unsigned int')}}

  uv = __builtin_elementwise_abs(uv);
  // expected-error@-1 {{1st argument must be a signed integer or floating point type (was 'unsigned4' (vector of 4 'unsigned int' values))}}
}

void test_builtin_elementwise_add_sat(int i, short s, double d, float4 v, int3 iv, unsigned3 uv, int *p) {
  i = __builtin_elementwise_add_sat(p, d);
  // expected-error@-1 {{arguments are of different types ('int *' vs 'double')}}

  struct Foo foo = __builtin_elementwise_add_sat(i, i);
  // expected-error@-1 {{initializing 'struct Foo' with an expression of incompatible type 'int'}}

  i = __builtin_elementwise_add_sat(i);
  // expected-error@-1 {{too few arguments to function call, expected 2, have 1}}

  i = __builtin_elementwise_add_sat();
  // expected-error@-1 {{too few arguments to function call, expected 2, have 0}}

  i = __builtin_elementwise_add_sat(i, i, i);
  // expected-error@-1 {{too many arguments to function call, expected 2, have 3}}

  i = __builtin_elementwise_add_sat(v, iv);
  // expected-error@-1 {{arguments are of different types ('float4' (vector of 4 'float' values) vs 'int3' (vector of 3 'int' values))}}

  i = __builtin_elementwise_add_sat(uv, iv);
  // expected-error@-1 {{arguments are of different types ('unsigned3' (vector of 3 'unsigned int' values) vs 'int3' (vector of 3 'int' values))}}

  v = __builtin_elementwise_add_sat(v, v);
  // expected-error@-1 {{1st argument must be a vector of integers (was 'float4' (vector of 4 'float' values))}}

  s = __builtin_elementwise_add_sat(i, s);

  enum e { one,
           two };
  i = __builtin_elementwise_add_sat(one, two);

  enum f { three };
  enum f x = __builtin_elementwise_add_sat(one, three);

  _BitInt(32) ext; // expected-warning {{'_BitInt' in C17 and earlier is a Clang extension}}
  ext = __builtin_elementwise_add_sat(ext, ext);

  const int ci;
  i = __builtin_elementwise_add_sat(ci, i);
  i = __builtin_elementwise_add_sat(i, ci);
  i = __builtin_elementwise_add_sat(ci, ci);

  i = __builtin_elementwise_add_sat(i, int_as_one); // ok (attributes don't match)?
  i = __builtin_elementwise_add_sat(i, b);          // ok (sugar doesn't match)?

  int A[10];
  A = __builtin_elementwise_add_sat(A, A);
  // expected-error@-1 {{1st argument must be a vector, integer or floating point type (was 'int *')}}

  int(ii);
  int j;
  j = __builtin_elementwise_add_sat(i, j);

  _Complex float c1, c2;
  c1 = __builtin_elementwise_add_sat(c1, c2);
  // expected-error@-1 {{1st argument must be a vector, integer or floating point type (was '_Complex float')}}
}

void test_builtin_elementwise_sub_sat(int i, short s, double d, float4 v, int3 iv, unsigned3 uv, int *p) {
  i = __builtin_elementwise_sub_sat(p, d);
  // expected-error@-1 {{arguments are of different types ('int *' vs 'double')}}

  struct Foo foo = __builtin_elementwise_sub_sat(i, i);
  // expected-error@-1 {{initializing 'struct Foo' with an expression of incompatible type 'int'}}

  i = __builtin_elementwise_sub_sat(i);
  // expected-error@-1 {{too few arguments to function call, expected 2, have 1}}

  i = __builtin_elementwise_sub_sat();
  // expected-error@-1 {{too few arguments to function call, expected 2, have 0}}

  i = __builtin_elementwise_sub_sat(i, i, i);
  // expected-error@-1 {{too many arguments to function call, expected 2, have 3}}

  i = __builtin_elementwise_sub_sat(v, iv);
  // expected-error@-1 {{arguments are of different types ('float4' (vector of 4 'float' values) vs 'int3' (vector of 3 'int' values))}}

  i = __builtin_elementwise_sub_sat(uv, iv);
  // expected-error@-1 {{arguments are of different types ('unsigned3' (vector of 3 'unsigned int' values) vs 'int3' (vector of 3 'int' values))}}

  v = __builtin_elementwise_sub_sat(v, v);
  // expected-error@-1 {{1st argument must be a vector of integers (was 'float4' (vector of 4 'float' values))}}

  s = __builtin_elementwise_sub_sat(i, s);

  enum e { one,
           two };
  i = __builtin_elementwise_sub_sat(one, two);

  enum f { three };
  enum f x = __builtin_elementwise_sub_sat(one, three);

  _BitInt(32) ext; // expected-warning {{'_BitInt' in C17 and earlier is a Clang extension}}
  ext = __builtin_elementwise_sub_sat(ext, ext);

  const int ci;
  i = __builtin_elementwise_sub_sat(ci, i);
  i = __builtin_elementwise_sub_sat(i, ci);
  i = __builtin_elementwise_sub_sat(ci, ci);

  i = __builtin_elementwise_sub_sat(i, int_as_one); // ok (attributes don't match)?
  i = __builtin_elementwise_sub_sat(i, b);          // ok (sugar doesn't match)?

  int A[10];
  A = __builtin_elementwise_sub_sat(A, A);
  // expected-error@-1 {{1st argument must be a vector, integer or floating point type (was 'int *')}}

  int(ii);
  int j;
  j = __builtin_elementwise_sub_sat(i, j);

  _Complex float c1, c2;
  c1 = __builtin_elementwise_sub_sat(c1, c2);
  // expected-error@-1 {{1st argument must be a vector, integer or floating point type (was '_Complex float')}}
}

void test_builtin_elementwise_max(int i, short s, double d, float4 v, int3 iv, unsigned3 uv, int *p) {
  i = __builtin_elementwise_max(p, d);
  // expected-error@-1 {{arguments are of different types ('int *' vs 'double')}}

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

  enum e { one,
           two };
  i = __builtin_elementwise_max(one, two);

  enum f { three };
  enum f x = __builtin_elementwise_max(one, three);

  _BitInt(32) ext; // expected-warning {{'_BitInt' in C17 and earlier is a Clang extension}}
  ext = __builtin_elementwise_max(ext, ext);

  const int ci;
  i = __builtin_elementwise_max(ci, i);
  i = __builtin_elementwise_max(i, ci);
  i = __builtin_elementwise_max(ci, ci);

  i = __builtin_elementwise_max(i, int_as_one); // ok (attributes don't match)?
  i = __builtin_elementwise_max(i, b);          // ok (sugar doesn't match)?

  int A[10];
  A = __builtin_elementwise_max(A, A);
  // expected-error@-1 {{1st argument must be a vector, integer or floating point type (was 'int *')}}

  int(ii);
  int j;
  j = __builtin_elementwise_max(i, j);

  _Complex float c1, c2;
  c1 = __builtin_elementwise_max(c1, c2);
  // expected-error@-1 {{1st argument must be a vector, integer or floating point type (was '_Complex float')}}
}

void test_builtin_elementwise_min(int i, short s, double d, float4 v, int3 iv, unsigned3 uv, int *p) {
  i = __builtin_elementwise_min(p, d);
  // expected-error@-1 {{arguments are of different types ('int *' vs 'double')}}

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

  enum e { one,
           two };
  i = __builtin_elementwise_min(one, two);

  enum f { three };
  enum f x = __builtin_elementwise_min(one, three);

  _BitInt(32) ext; // expected-warning {{'_BitInt' in C17 and earlier is a Clang extension}}
  ext = __builtin_elementwise_min(ext, ext);

  const int ci;
  i = __builtin_elementwise_min(ci, i);
  i = __builtin_elementwise_min(i, ci);
  i = __builtin_elementwise_min(ci, ci);

  i = __builtin_elementwise_min(i, int_as_one); // ok (attributes don't match)?
  i = __builtin_elementwise_min(i, b);          // ok (sugar doesn't match)?

  int A[10];
  A = __builtin_elementwise_min(A, A);
  // expected-error@-1 {{1st argument must be a vector, integer or floating point type (was 'int *')}}

  int(ii);
  int j;
  j = __builtin_elementwise_min(i, j);

  _Complex float c1, c2;
  c1 = __builtin_elementwise_min(c1, c2);
  // expected-error@-1 {{1st argument must be a vector, integer or floating point type (was '_Complex float')}}
}

void test_builtin_elementwise_ceil(int i, float f, double d, float4 v, int3 iv, unsigned u, unsigned4 uv) {

  struct Foo s = __builtin_elementwise_ceil(f);
  // expected-error@-1 {{initializing 'struct Foo' with an expression of incompatible type 'float'}}

  i = __builtin_elementwise_ceil();
  // expected-error@-1 {{too few arguments to function call, expected 1, have 0}}

  i = __builtin_elementwise_ceil(i);
  // expected-error@-1 {{1st argument must be a floating point type (was 'int')}}

  i = __builtin_elementwise_ceil(f, f);
  // expected-error@-1 {{too many arguments to function call, expected 1, have 2}}

  u = __builtin_elementwise_ceil(u);
  // expected-error@-1 {{1st argument must be a floating point type (was 'unsigned int')}}

  uv = __builtin_elementwise_ceil(uv);
  // expected-error@-1 {{1st argument must be a floating point type (was 'unsigned4' (vector of 4 'unsigned int' values))}}
}

void test_builtin_elementwise_cos(int i, float f, double d, float4 v, int3 iv, unsigned u, unsigned4 uv) {

  struct Foo s = __builtin_elementwise_cos(f);
  // expected-error@-1 {{initializing 'struct Foo' with an expression of incompatible type 'float'}}

  i = __builtin_elementwise_cos();
  // expected-error@-1 {{too few arguments to function call, expected 1, have 0}}

  i = __builtin_elementwise_cos(i);
  // expected-error@-1 {{1st argument must be a floating point type (was 'int')}}

  i = __builtin_elementwise_cos(f, f);
  // expected-error@-1 {{too many arguments to function call, expected 1, have 2}}

  u = __builtin_elementwise_cos(u);
  // expected-error@-1 {{1st argument must be a floating point type (was 'unsigned int')}}

  uv = __builtin_elementwise_cos(uv);
  // expected-error@-1 {{1st argument must be a floating point type (was 'unsigned4' (vector of 4 'unsigned int' values))}}
}

void test_builtin_elementwise_floor(int i, float f, double d, float4 v, int3 iv, unsigned u, unsigned4 uv) {

  struct Foo s = __builtin_elementwise_floor(f);
  // expected-error@-1 {{initializing 'struct Foo' with an expression of incompatible type 'float'}}

  i = __builtin_elementwise_floor();
  // expected-error@-1 {{too few arguments to function call, expected 1, have 0}}

  i = __builtin_elementwise_floor(i);
  // expected-error@-1 {{1st argument must be a floating point type (was 'int')}}

  i = __builtin_elementwise_floor(f, f);
  // expected-error@-1 {{too many arguments to function call, expected 1, have 2}}

  u = __builtin_elementwise_floor(u);
  // expected-error@-1 {{1st argument must be a floating point type (was 'unsigned int')}}

  uv = __builtin_elementwise_floor(uv);
  // expected-error@-1 {{1st argument must be a floating point type (was 'unsigned4' (vector of 4 'unsigned int' values))}}
}

void test_builtin_elementwise_roundeven(int i, float f, double d, float4 v, int3 iv, unsigned u, unsigned4 uv) {

  struct Foo s = __builtin_elementwise_roundeven(f);
  // expected-error@-1 {{initializing 'struct Foo' with an expression of incompatible type 'float'}}

  i = __builtin_elementwise_roundeven();
  // expected-error@-1 {{too few arguments to function call, expected 1, have 0}}

  i = __builtin_elementwise_roundeven(i);
  // expected-error@-1 {{1st argument must be a floating point type (was 'int')}}

  i = __builtin_elementwise_roundeven(f, f);
  // expected-error@-1 {{too many arguments to function call, expected 1, have 2}}

  u = __builtin_elementwise_roundeven(u);
  // expected-error@-1 {{1st argument must be a floating point type (was 'unsigned int')}}

  uv = __builtin_elementwise_roundeven(uv);
  // expected-error@-1 {{1st argument must be a floating point type (was 'unsigned4' (vector of 4 'unsigned int' values))}}
}

void test_builtin_elementwise_sin(int i, float f, double d, float4 v, int3 iv, unsigned u, unsigned4 uv) {

  struct Foo s = __builtin_elementwise_sin(f);
  // expected-error@-1 {{initializing 'struct Foo' with an expression of incompatible type 'float'}}

  i = __builtin_elementwise_sin();
  // expected-error@-1 {{too few arguments to function call, expected 1, have 0}}

  i = __builtin_elementwise_sin(i);
  // expected-error@-1 {{1st argument must be a floating point type (was 'int')}}

  i = __builtin_elementwise_sin(f, f);
  // expected-error@-1 {{too many arguments to function call, expected 1, have 2}}

  u = __builtin_elementwise_sin(u);
  // expected-error@-1 {{1st argument must be a floating point type (was 'unsigned int')}}

  uv = __builtin_elementwise_sin(uv);
  // expected-error@-1 {{1st argument must be a floating point type (was 'unsigned4' (vector of 4 'unsigned int' values))}}
}

void test_builtin_elementwise_trunc(int i, float f, double d, float4 v, int3 iv, unsigned u, unsigned4 uv) {

  struct Foo s = __builtin_elementwise_trunc(f);
  // expected-error@-1 {{initializing 'struct Foo' with an expression of incompatible type 'float'}}

  i = __builtin_elementwise_trunc();
  // expected-error@-1 {{too few arguments to function call, expected 1, have 0}}

  i = __builtin_elementwise_trunc(i);
  // expected-error@-1 {{1st argument must be a floating point type (was 'int')}}

  i = __builtin_elementwise_trunc(f, f);
  // expected-error@-1 {{too many arguments to function call, expected 1, have 2}}

  u = __builtin_elementwise_trunc(u);
  // expected-error@-1 {{1st argument must be a floating point type (was 'unsigned int')}}

  uv = __builtin_elementwise_trunc(uv);
  // expected-error@-1 {{1st argument must be a floating point type (was 'unsigned4' (vector of 4 'unsigned int' values))}}
}
