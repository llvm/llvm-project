// RUN: %clang_cc1 -Wpedantic-macros %s -fsyntax-only -verify

// This test verifies that the -Wpedantic-macros warnings don't fire in the
// annotation pragmas themselves. This ensures you can annotate macros with
// more than one of the pragmas without spewing warnings all over the code.

#include "Inputs/pedantic-macro-interplay.h"

#define UNSAFE_MACRO_2 1
#pragma clang deprecated(UNSAFE_MACRO_2, "Don't use this!")
// not-expected-warning@+1{{macro 'UNSAFE_MACRO_2' has been marked as deprecated: Don't use this!}}
#pragma clang restrict_expansion(UNSAFE_MACRO_2, "Don't use this!")

// expected-no-diagnostics
