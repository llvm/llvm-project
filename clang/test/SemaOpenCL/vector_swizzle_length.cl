// RUN: %clang_cc1 -x cl %s -verify -pedantic -fsyntax-only

typedef float float8 __attribute__((ext_vector_type(8)));

void foo(void) {
    float8 f2 = (float8)(0, 0, 0, 0, 0, 0, 0, 0);

    f2.s01234; // expected-error {{vector component access has invalid length 5; supported lengths are: 1,2,3,4,8,16}}
    f2.xyzxy; // expected-error {{vector component access has invalid length 5; supported lengths are: 1,2,3,4,8,16}}
}
