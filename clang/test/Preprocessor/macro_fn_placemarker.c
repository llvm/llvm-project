// RUN: %clang_cc1 %s -E | grep 'foo(A, )'

#define X(Y) foo(A, Y)
X()

