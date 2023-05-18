// RUN: %clang -E -o %t -C %s
// RUN: grep '^int x; // comment' %t
// RUN: grep '^x x' %t
// RUN: %clang -E -o %t -CC %s
// RUN: grep '^int x; // comment' %t
// RUN: grep '^x /\* comment \*/ x /\* comment \*/' %t

int x; // comment

#define A(foo, bar) foo bar
#define B x // comment 

A(B, B)

