// RUN: %check_clang_tidy %s misc-include-cleaner %t

#define I 42
void f(void) { I; }

#define H 42
void g(void) { H; }
