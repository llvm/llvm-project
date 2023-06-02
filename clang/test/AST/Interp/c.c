// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -verify %s
// RUN: %clang_cc1 -verify=ref %s

/// expected-no-diagnostics
/// ref-no-diagnostics

_Static_assert(1, "");
_Static_assert(0 != 1, "");
_Static_assert(1.0 == 1.0, "");
_Static_assert( (5 > 4) + (3 > 2) == 2, "");

int a = (1 == 1 ? 5 : 3);
