// RUN: %clang -E -x c-header %s > %t
// RUN: grep 'B B' %t

#define A B
A A

