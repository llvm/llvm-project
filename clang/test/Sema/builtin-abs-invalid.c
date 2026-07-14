// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fsyntax-only -verify %s
// expected-no-diagnostics

int abs(int);

typedef int int1 __attribute__((__vector_size__(4)));

void test_vector_abs(int1 x) {
    (void)abs(x);
}