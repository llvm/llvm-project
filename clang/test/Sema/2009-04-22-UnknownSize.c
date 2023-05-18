// RUN: not %clang_cc1 -fsyntax-only -verify %s
// PR2958
static struct foo s; // expected-error { tentative definition has type 'struct foo' that is never completed }
struct foo *p = &s;
