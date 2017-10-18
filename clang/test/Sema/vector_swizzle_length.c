// RUN: %clang_cc1 -x c %s -verify -pedantic -fsyntax-only
// expected-no-diagnostics

typedef float float8 __attribute__((ext_vector_type(8)));

void foo() {
    float8 f2 = (float8){0, 0, 0, 0, 0, 0, 0, 0};
    (void)f2.s01234;
    (void)f2.xyzxy;
}
