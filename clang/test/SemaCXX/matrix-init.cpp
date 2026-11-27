// RUN: %clang_cc1 -fenable-matrix -fsyntax-only -verify=expected %s

// A braced list cannot initialize a matrix: unlike C, where the elements used to
// be diagnosed as excess initializers (a warning) and the matrix silently
// zero-initialized, C++ already rejected them. The wording is now the same as
// for every other type that has no initializer-list form.
// https://github.com/llvm/llvm-project/issues/246536

typedef int m31 __attribute__((matrix_type(3, 1)));

m31 global = {1, 2, 3}; // expected-error {{illegal initializer type}}

void one_element() {
  m31 a = {1}; // expected-error {{illegal initializer type}}
}

void array_of_matrices() {
  m31 arr[2] = {{1, 2, 3}, {4, 5, 6}}; // expected-error 2 {{illegal initializer type}}
}

struct S {
  m31 m;
  char c;
};
S s = {{7, 8, 9}, 'x'}; // expected-error {{illegal initializer type}}

void empty_list() {
  m31 a = {};
  (void)a;
}
