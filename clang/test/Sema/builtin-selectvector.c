// RUN: %clang_cc1 -std=c99 %s -pedantic -verify -triple=x86_64-apple-darwin9

typedef double double2 __attribute__((ext_vector_type(2)));
typedef double double4 __attribute__((ext_vector_type(4)));

typedef _Bool bool2 __attribute__((ext_vector_type(2)));
typedef _Bool bool4 __attribute__((ext_vector_type(4)));

void test(bool2 vec_bool2, bool4 vec_bool4, double2 vec_double2, double4 vec_double4) {
  __builtin_selectvector(); // expected-error {{too few arguments to function call, expected 3, have 0}}
  (void)__builtin_selectvector(0, 0, 0); // expected-error {{1st argument must be a vector type (was 'int')}}
  (void)__builtin_selectvector(vec_double2, 0, 0); // expected-error {{arguments are of different types ('double2' (vector of 2 'double' values) vs 'int')}}
  (void)__builtin_selectvector(vec_double2, vec_double2, 0); // expected-error {{3rd argument must be a vector of bools (was 'int')}}
  (void)__builtin_selectvector(vec_double2, vec_double2, vec_double2); // expected-error {{3rd argument must be a vector of bools (was 'double2' (vector of 2 'double' values))}}
  (void)__builtin_selectvector(vec_double2, vec_double4, vec_bool2); // expected-error {{arguments are of different types ('double2' (vector of 2 'double' values) vs 'double4' (vector of 4 'double' values))}}
  (void)__builtin_selectvector(vec_double2, vec_double2, vec_bool4); // expected-error {{vector operands do not have the same number of elements ('double2' (vector of 2 'double' values) and 'bool4' (vector of 4 '_Bool' values))}}
  (void)__builtin_selectvector(vec_double2, vec_double2, vec_bool2);
}
