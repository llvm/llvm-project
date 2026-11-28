// RUN: %clang_cc1 -fenable-matrix %s -verify -fexperimental-new-constant-interpreter
// RUN: %clang_cc1 -fenable-matrix %s -verify

typedef float fx2x2_t __attribute__((matrix_type(2, 2)));
// A braced list is rejected before it can reach either interpreter, so the
// matrix is no longer silently zero-initialized here.
fx2x2_t ret_matrix() { return (fx2x2_t){1.0f, 2.0f, 3.0f, 4.0f}; } // expected-error {{illegal initializer type}}
