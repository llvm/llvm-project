// RUN: %clang_cc1 -std=c++20 -emit-pch %s -o %t
// RUN: %clang_cc1 -std=c++20 -include-pch %t -verify %s
// expected-no-diagnostics

#ifndef HEADER_INCLUDED
#define HEADER_INCLUDED

consteval int immediate();
int regular_function() {
    return 0;
}

struct S {
  int a = immediate() + regular_function();
};

int f(int arg = immediate()) {
    return arg;
}

#else

consteval int immediate() {
    return 0;
}

void test() {
    f(0);
    f();
    S s{0};
    S t{0};
}

#endif
