// RUN: %clang_cc1 -fsyntax-only -std=c23 -verify %s

#define H1(X, ...)       X __VA_OPT__(##) __VA_ARGS__ // expected-error {{'##' cannot appear at start of __VA_OPT__ argument}}

