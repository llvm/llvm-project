// RUN: %clang_cc1 -fsyntax-only -verify -fenable-matrix %s

typedef double double4x4 __attribute__((matrix_type(4, 4)));
typedef unsigned u4x4 __attribute__((matrix_type(4, 4)));

__attribute__((objc_root_class))
@interface MatrixValue
@property double4x4 value;
@end

void test_element_type_mismatch(u4x4 m, MatrixValue *mv) {
  m = __builtin_matrix_transpose(mv.value);
  // expected-error@-1 {{assigning to 'u4x4' (aka 'unsigned int __attribute__((matrix_type(4, 4)))') from incompatible type 'double __attribute__((matrix_type(4, 4)))'}}
}

typedef double double3x3 __attribute__((matrix_type(3, 3)));

double test_dimension_mismatch(double3x3 m, MatrixValue *mv) {
  m = __builtin_matrix_transpose(mv.value);
  // expected-error@-1 {{assigning to 'double3x3' (aka 'double __attribute__((matrix_type(3, 3)))') from incompatible type 'double __attribute__((matrix_type(4, 4)))'}}
}
