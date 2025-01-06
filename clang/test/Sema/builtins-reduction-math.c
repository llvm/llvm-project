// RUN: %clang_cc1 %s -pedantic -verify -triple=x86_64-apple-darwin9

typedef float float4 __attribute__((ext_vector_type(4)));
typedef int int3 __attribute__((ext_vector_type(3)));
typedef unsigned unsigned4 __attribute__((ext_vector_type(4)));

struct Foo {
  char *p;
};

void test_builtin_reduce_max(int i, float4 v, int3 iv) {
  struct Foo s = __builtin_reduce_max(iv);
  // expected-error@-1 {{initializing 'struct Foo' with an expression of incompatible type 'int'}}

  i = __builtin_reduce_max(v, v);
  // expected-error@-1 {{too many arguments to function call, expected 1, have 2}}

  i = __builtin_reduce_max();
  // expected-error@-1 {{too few arguments to function call, expected 1, have 0}}

  i = __builtin_reduce_max(i);
  // expected-error@-1 {{1st argument must be a vector type (was 'int')}}
}

void test_builtin_reduce_min(int i, float4 v, int3 iv) {
  struct Foo s = __builtin_reduce_min(iv);
  // expected-error@-1 {{initializing 'struct Foo' with an expression of incompatible type 'int'}}

  i = __builtin_reduce_min(v, v);
  // expected-error@-1 {{too many arguments to function call, expected 1, have 2}}

  i = __builtin_reduce_min();
  // expected-error@-1 {{too few arguments to function call, expected 1, have 0}}

  i = __builtin_reduce_min(i);
  // expected-error@-1 {{1st argument must be a vector type (was 'int')}}
}

void test_builtin_reduce_add(int i, float4 v, int3 iv) {
  struct Foo s = __builtin_reduce_add(iv);
  // expected-error@-1 {{initializing 'struct Foo' with an expression of incompatible type 'int'}}

  i = __builtin_reduce_add();
  // expected-error@-1 {{too few arguments to function call, expected 1, have 0}}

  i = __builtin_reduce_add(iv, iv);
  // expected-error@-1 {{too many arguments to function call, expected 1, have 2}}

  i = __builtin_reduce_add(i);
  // expected-error@-1 {{1st argument must be a vector of integers (was 'int')}}

  i = __builtin_reduce_add(v);
  // expected-error@-1 {{1st argument must be a vector of integers (was 'float4' (vector of 4 'float' values))}}
}

void test_builtin_reduce_mul(int i, float4 v, int3 iv) {
  struct Foo s = __builtin_reduce_mul(iv);
  // expected-error@-1 {{initializing 'struct Foo' with an expression of incompatible type 'int'}}

  i = __builtin_reduce_mul();
  // expected-error@-1 {{too few arguments to function call, expected 1, have 0}}

  i = __builtin_reduce_mul(iv, iv);
  // expected-error@-1 {{too many arguments to function call, expected 1, have 2}}

  i = __builtin_reduce_mul(i);
  // expected-error@-1 {{1st argument must be a vector of integers (was 'int')}}

  i = __builtin_reduce_mul(v);
  // expected-error@-1 {{1st argument must be a vector of integers (was 'float4' (vector of 4 'float' values))}}
}

void test_builtin_reduce_xor(int i, float4 v, int3 iv) {
  struct Foo s = __builtin_reduce_xor(iv);
  // expected-error@-1 {{initializing 'struct Foo' with an expression of incompatible type 'int'}}

  i = __builtin_reduce_xor();
  // expected-error@-1 {{too few arguments to function call, expected 1, have 0}}

  i = __builtin_reduce_xor(iv, iv);
  // expected-error@-1 {{too many arguments to function call, expected 1, have 2}}

  i = __builtin_reduce_xor(i);
  // expected-error@-1 {{1st argument must be a vector of integers (was 'int')}}

  i = __builtin_reduce_xor(v);
  // expected-error@-1 {{1st argument must be a vector of integers (was 'float4' (vector of 4 'float' values))}}
}

void test_builtin_reduce_or(int i, float4 v, int3 iv) {
  struct Foo s = __builtin_reduce_or(iv);
  // expected-error@-1 {{initializing 'struct Foo' with an expression of incompatible type 'int'}}

  i = __builtin_reduce_or();
  // expected-error@-1 {{too few arguments to function call, expected 1, have 0}}

  i = __builtin_reduce_or(iv, iv);
  // expected-error@-1 {{too many arguments to function call, expected 1, have 2}}

  i = __builtin_reduce_or(i);
  // expected-error@-1 {{1st argument must be a vector of integers (was 'int')}}

  i = __builtin_reduce_or(v);
  // expected-error@-1 {{1st argument must be a vector of integers (was 'float4' (vector of 4 'float' values))}}
}

void test_builtin_reduce_and(int i, float4 v, int3 iv) {
  struct Foo s = __builtin_reduce_and(iv);
  // expected-error@-1 {{initializing 'struct Foo' with an expression of incompatible type 'int'}}

  i = __builtin_reduce_and();
  // expected-error@-1 {{too few arguments to function call, expected 1, have 0}}

  i = __builtin_reduce_and(iv, iv);
  // expected-error@-1 {{too many arguments to function call, expected 1, have 2}}

  i = __builtin_reduce_and(i);
  // expected-error@-1 {{1st argument must be a vector of integers (was 'int')}}

  i = __builtin_reduce_and(v);
  // expected-error@-1 {{1st argument must be a vector of integers (was 'float4' (vector of 4 'float' values))}}
}

void test_builtin_reduce_maximum(int i, float4 v, int3 iv) {
  struct Foo s = __builtin_reduce_maximum(v);
  // expected-error@-1 {{initializing 'struct Foo' with an expression of incompatible type 'float'}}

  i = __builtin_reduce_maximum(v, v);
  // expected-error@-1 {{too many arguments to function call, expected 1, have 2}}

  i = __builtin_reduce_maximum();
  // expected-error@-1 {{too few arguments to function call, expected 1, have 0}}

  i = __builtin_reduce_maximum(i);
  // expected-error@-1 {{1st argument must be a vector of floating points (was 'int')}}
}

void test_builtin_reduce_minimum(int i, float4 v, int3 iv) {
  struct Foo s = __builtin_reduce_minimum(v);
  // expected-error@-1 {{initializing 'struct Foo' with an expression of incompatible type 'float'}}

  i = __builtin_reduce_minimum(v, v);
  // expected-error@-1 {{too many arguments to function call, expected 1, have 2}}

  i = __builtin_reduce_minimum();
  // expected-error@-1 {{too few arguments to function call, expected 1, have 0}}

  i = __builtin_reduce_minimum(i);
  // expected-error@-1 {{1st argument must be a vector of floating points (was 'int')}}
}
