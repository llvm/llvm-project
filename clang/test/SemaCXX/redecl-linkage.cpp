// RUN: %clang_cc1 -std=c++17 %s -verify

static int k = 3;
extern int k; // expected-error {{redeclaration of k with different linkage}}

