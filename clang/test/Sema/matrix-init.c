// RUN: %clang_cc1 -fenable-matrix -fsyntax-only -verify=expected %s

// Matrix types have no initializer syntax outside of HLSL yet, so a braced list
// must be rejected instead of being accepted and silently zero-initializing the
// matrix. The elements used to fall through to the "excess elements" diagnostic,
// which is only a warning in C.
// https://github.com/llvm/llvm-project/issues/246536

typedef int m31 __attribute__((matrix_type(3, 1)));
typedef float fx2x2 __attribute__((matrix_type(2, 2)));

m31 global = {1, 2, 3}; // expected-error {{illegal initializer type}}

// A compound literal is rejected the same way.
m31 make_compound_literal(void) { return (m31){1, 2, 3}; } // expected-error {{illegal initializer type}}

// So is a single element: there is no splat form either.
void one_element(void) {
  m31 a = {1}; // expected-error {{illegal initializer type}}
  (void)a;
}

// Each matrix in an array gets its own diagnostic.
void array_of_matrices(void) {
  m31 arr[2] = {{1, 2, 3}, {4, 5, 6}}; // expected-error 2 {{illegal initializer type}}
  (void)arr;
}

// The initializer of a matrix member used to be diagnosed as a warning here and
// then crash the constant evaluator.
void struct_member(void) {
  struct S {
    m31 m;
    char c;
  };
  struct S s = {{7, 8, 9}, 'x'}; // expected-error {{illegal initializer type}}
  (void)s;
}

// An empty list is still accepted: it value-initializes the matrix.
void empty_list(void) {
  m31 a = {};
  fx2x2 b = {};
  (void)a;
  (void)b;
}
