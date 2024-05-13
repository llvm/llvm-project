// RUN: %clang_cc1 -fsyntax-only -verify %s
char test1[1]="f"; // expected-error {{initializer-string for char array is too long, array size is 1 but initializer has size 2 (including the null terminating character)}}
char test2[1]="";
