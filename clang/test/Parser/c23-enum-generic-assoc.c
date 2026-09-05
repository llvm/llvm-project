// RUN: %clang_cc1 -fsyntax-only -verify -std=c23 %s
// RUN: %clang_cc1 -fsyntax-only -verify -x c++ %s

typedef long l;
enum E : long { X };


#ifdef __cplusplus

static_assert(_Generic(0L, enum A : l { B } : 1, int: 0), ""); // expected-error {{'A' cannot be defined in a type specifier}}

#else

static_assert(_Generic(0L, enum E : long { X } : 1, int: 0), ""); // expected-no-diagnostics
static_assert(_Generic(0L, enum E : 1, int: 0), ""); // expected-no-diagnostics
static_assert(_Generic(0L, enum A : l { B } : 1, int: 0), ""); // expected-no-diagnostics
static_assert(_Generic(0L, struct { enum A : l { B } a : 1; } : 0, long: 1), ""); // expected-no-diagnostics

#endif