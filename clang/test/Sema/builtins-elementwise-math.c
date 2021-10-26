// RUN: %clang_cc1 -std=c99 %s -pedantic -verify -triple=x86_64-apple-darwin9

typedef float float4 __attribute__((ext_vector_type(4)));
typedef int int3 __attribute__((ext_vector_type(3)));

struct Foo {
  char *p;
};

__attribute__((address_space(1))) int int_as_one;
typedef int bar;
bar b;

void test_builtin_elementwise_max(int i, short s, double d, float4 v, int3 iv, int *p) {
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

  s = __builtin_elementwise_max(i, s);

  enum e { one,
           two };
  i = __builtin_elementwise_max(one, two);

  enum f { three };
  enum f x = __builtin_elementwise_max(one, three);

  _ExtInt(32) ext;
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

void test_builtin_elementwise_min(int i, short s, double d, float4 v, int3 iv, int *p) {
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

  s = __builtin_elementwise_min(i, s);

  enum e { one,
           two };
  i = __builtin_elementwise_min(one, two);

  enum f { three };
  enum f x = __builtin_elementwise_min(one, three);

  _ExtInt(32) ext;
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
