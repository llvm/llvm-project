// RUN: %clang_cc1 -fsyntax-only -verify -triple x86_64-unknown-linux %s

typedef int int4 __attribute__((ext_vector_type(4)));
typedef short short4 __attribute__((ext_vector_type(4)));
typedef int int3 __attribute__((ext_vector_type(3)));
typedef _Bool bool4 __attribute__((ext_vector_type(4)));

void test_valid(int x, int4 v, bool4 bv, _Bool b, char c) {
  short s = __builtin_elementwise_saturating_cast(x, short);
  short4 sv = __builtin_elementwise_saturating_cast(v, short);
  short4 sv2 = __builtin_elementwise_saturating_cast(v, short4);
  bool4 bsv = __builtin_elementwise_saturating_cast(v, bool4);
  short4 sbv = __builtin_elementwise_saturating_cast(bv, short);

  (void)__builtin_elementwise_saturating_cast(b, short);
  (void)__builtin_elementwise_saturating_cast(c, short);
  enum E { e };
  (void)__builtin_elementwise_saturating_cast(e, short);
  (void)__builtin_elementwise_saturating_cast(x, _Bool);
  (void)__builtin_elementwise_saturating_cast(x, const int);
}

void test_invalid(int x, int4 v, float f) {
  (void)__builtin_elementwise_saturating_cast(x, float);
  // expected-error@-1 {{2nd argument must be a scalar or vector of integer types (was 'float')}}

  (void)__builtin_elementwise_saturating_cast(f, short);
  // expected-error@-1 {{1st argument must be a scalar or vector of integer types (was 'float')}}

  (void)__builtin_elementwise_saturating_cast(x, short4);
  // expected-error@-1 {{first two arguments to '__builtin_elementwise_saturating_cast' must be vectors}}

  (void)__builtin_elementwise_saturating_cast((int3){1, 2, 3}, short4);
  // expected-error@-1 {{vector operands do not have the same number of elements}}

  (void)__builtin_elementwise_saturating_cast(x, short, x);
  // expected-error@-1 {{expected ')'}}

}
