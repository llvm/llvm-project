// RUN: %clang_cc1  -std=gnu99 -fsyntax-only -verify %s
// expected-no-diagnostics

void f0(restrict id a0) {}

void f1(restrict id *a0) {}

void f2(restrict Class a0) {}

void f3(restrict Class *a0) {}
