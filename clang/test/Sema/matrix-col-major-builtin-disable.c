// RUN: %clang_cc1 %s -fenable-matrix -fmatrix-memory-layout=row-major -pedantic -verify -triple=x86_64-apple-darwin9

typedef float sx5x10_t __attribute__((matrix_type(5, 10)));

void column_major_load(float *p1) {
  sx5x10_t a1 = __builtin_matrix_column_major_load(p1, 5, 11, 5);
  // expected-error@-1 {{matrix column major load is not supported with -fmatrix-memory-layout=row-major. Pass -fmatrix-memory-layout=column-major to enable it}}
}

void column_major_store(sx5x10_t *m1, float *p1) {
  __builtin_matrix_column_major_store(*m1, p1, 1);
  // expected-error@-1 {{matrix column major store is not supported with -fmatrix-memory-layout=row-major. Pass -fmatrix-memory-layout=column-major to enable it}}
}
