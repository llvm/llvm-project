// RUN: %clang_cc1 -fsyntax-only -verify %s

int *test1(int *a)         { return a + 1; }
int *test2(int *a)         { return 1 + a; }
int *test3(int *a)         { return a - 1; }
int  test4(int *a, int *b) { return a - b; }

int  test5(int *a, int *b) { return a + b; } /* expected-error {{invalid operands}} */
int *test6(int *a)         { return 1 - a; } /* expected-error {{invalid operands}} */
