// RUN: %clang_cc1 -fsyntax-only -verify %s

void func(int n) {
    int grp[n][n];
    int (*ptr)[n];

    for (int i = 0; i < n; i++)
        ptr = &grp[i]; // expected-error {{incompatible pointer types assigning to 'int (*)[n]' from 'int (*)[n]'; VLA types differ despite using the same array size expression}}
}

void func(int n, int (&array)[n]) {
  int (&other)[n] = array; // expected-error {{non-const lvalue reference to type 'int[n]' cannot bind to a value of unrelated type 'int[n]'; VLA types differ despite using the same array size expression}}
}
