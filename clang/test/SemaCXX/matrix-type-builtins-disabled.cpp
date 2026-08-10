// RUN: %clang_cc1 %s -pedantic -std=c++11 -verify -triple=x86_64-apple-darwin9

// Make sure we fail without -fenable-matrix when
// __builtin_matrix_column_major_load is used to construct a new matrix type.
void column_major_load_with_stride(int *Ptr) {
  auto m = __builtin_matrix_column_major_load(Ptr, 2, 2, 2);
  // expected-error@-1 {{matrix types extension is disabled. Pass -fenable-matrix to enable it}}
}
