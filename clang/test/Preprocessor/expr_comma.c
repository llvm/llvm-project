// Comma is not allowed in C89
// RUN: not %clang_cc1 -E %s -std=c89 -pedantic-errors

// Comma is allowed if unevaluated in C99
// RUN: %clang_cc1 -E %s -std=c99 -pedantic-errors 

// PR2279

#if 0? 1,2:3
#endif
