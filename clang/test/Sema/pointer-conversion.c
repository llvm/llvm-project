//RUN: %clang_cc1 -fsyntax-only -verify %s

char * c;
char const ** c2 = &c; // expected-warning {{discards qualifiers in nested pointer types}}

typedef char dchar;
dchar *** c3 = &c2; // expected-warning {{discards qualifiers in nested pointer types}}

volatile char * c4;
char ** c5 = &c4; // expected-warning {{discards qualifiers in nested pointer types}}
