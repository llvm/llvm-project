// RUN: %clang_cc1 %s -verify -pedantic -fsyntax-only
// expected-no-diagnostics

typedef int int3 __attribute__((ext_vector_type(3)));

void test(void)
{
    int index = (int3)(1, 2, 3).x * (int3)(3, 2, 1).y;
}

