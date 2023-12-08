
// RUN: %clang_cc1 -E %s | grep 'foo{a, b, c, d, e}'
// RUN: %clang_cc1 -E %s | grep 'foo2{d, C, B}'
// RUN: %clang_cc1 -E %s | grep 'foo2{d,e, C, B}'

#define va1(...) foo{a, __VA_ARGS__, e}
va1(b, c, d)
#define va2(a, b, ...) foo2{__VA_ARGS__, b, a}
va2(B, C, d)
va2(B, C, d,e)

