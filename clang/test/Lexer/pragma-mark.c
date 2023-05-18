// RUN: %clang_cc1 %s -fsyntax-only -verify
// expected-no-diagnostics

// Lexer diagnostics shouldn't be included in #pragma mark.
#pragma mark Mike's world
_Pragma("mark foo ' bar")

#define X(S) _Pragma(S)
X("mark foo ' bar")

int i;

