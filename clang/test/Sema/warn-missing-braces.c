// RUN: %clang_cc1 -fsyntax-only -Wmissing-braces -verify %s

int a[2][2] = { 0, 1, 2, 3 }; // expected-warning{{suggest braces}} expected-warning{{suggest braces}}
