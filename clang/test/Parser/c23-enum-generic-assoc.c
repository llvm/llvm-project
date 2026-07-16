// RUN: %clang_cc1 -fsyntax-only -verify -std=c23 %s
// expected-no-diagnostics

typedef long l;

_Static_assert(_Generic(0L, enum E : long { A } : 0, int: 1) == 0, "");

_Static_assert(_Generic(0L, enum E : 0, int: 1) == 0, "");

_Static_assert(_Generic(0L, enum A : l { B } : 0, int: 1) == 0, "");