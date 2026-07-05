// RUN: %clang_cc1 %s -verify -pedantic -fsyntax-only
// expected-no-diagnostics

kernel void foo(global int a[], local int b[], constant int c[4]) { }

void bar(global int a[], local int b[], constant int c[4], int d[]) { }
