// RUN: %clang_cc1 -std=c++11 -Wuninitialized -verify %s
// expected-no-diagnostics

int test1(int x) {
    int y;
    asm goto("# %0 %1 %2" : "=r"(y) : "r"(x) : : err);
    return y;
  err:
    return -1;
}
