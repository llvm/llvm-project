// RUN: %clang_cc1 -fsyntax-only -verify -fenable-matrix %s

struct Foo {};
__attribute__((objc_root_class))
@interface FooValue
@property struct Foo value;
@end

typedef double double4x4 __attribute__((matrix_type(4, 4)));

// Check that we generate proper error messages for invalid placeholder types.
//
double test_index_placeholders(double4x4 m, FooValue *iv) {
  return m[iv.value][iv.value];
  // expected-error@-1 {{matrix row index is not an integer}}
  // expected-error@-2 {{matrix column index is not an integer}}
}

double test_base_and_index_placeholders(FooValue *m, FooValue *iv) {
  return m.value[iv.value][iv.value];
  // expected-error@-1 {{subscripted value is not an array, pointer, or vector}}
}
