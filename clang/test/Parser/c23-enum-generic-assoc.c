// RUN: %clang_cc1 -fsyntax-only -verify -std=c23 %s
// expected-no-diagnostics

typedef long l;

_Static_assert(_Generic(0L, enum E : long { A } : 1, int: 0), "");

_Static_assert(_Generic(0L, enum E : 1, int: 0), "");

_Static_assert(_Generic(0L, enum A : l { B } : 1, int: 0), "");

_Static_assert(_Generic(0L, struct { enum A : l { B } a : 1; } : 0, long: 1), "");
