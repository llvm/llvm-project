// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics
typedef struct {
        int a; int : 24; char b;
} S;

S a = { 1, 2 };
