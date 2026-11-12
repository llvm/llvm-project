// RUN: %clang_cc1 %s -triple i686-apple-darwin -verify -fsyntax-only

// Matrix types are disabled by default.

#if __has_extension(matrix_types)
#error Expected extension 'matrix_types' to be disabled
#endif

typedef double dx5x5_t __attribute__((matrix_type(5, 5)));
// expected-error@-1 {{matrix types extension is disabled. Pass -fenable-matrix to enable it}}

void load_store_double(dx5x5_t *a, dx5x5_t *b) {
  *a = *b;
}
