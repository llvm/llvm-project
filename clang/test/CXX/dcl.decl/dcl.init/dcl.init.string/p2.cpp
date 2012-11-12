// RUN: %clang_cc1 -fsyntax-only -verify %s
char test1[1]="f"; // expected-error {{initializer-string for char array is too long}}
