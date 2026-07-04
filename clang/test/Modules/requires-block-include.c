// Tests that a header which only belongs to a module under an unsatisfied
// `requires` block falls back to a plain textual #include, rather than being
// translated into a module import. This is the motivating use case: a header
// can opt into being modular only under, e.g., `requires cplusplus`, while
// remaining an ordinary textual include otherwise.
//
// RUN: rm -rf %t
// RUN: split-file %s %t
//
// Compiled as C: the 'cxxonly' module is guarded by `requires cplusplus` and so
// does not exist. Its header never enters the Headers map, so the #include is a
// plain textual include (no include-to-import translation) and succeeds.
// -Rmodule-include-translation would emit a remark if any translation happened;
// expected-no-diagnostics asserts none does.
// RUN: %clang_cc1 -x c -std=c99 -fmodules -fimplicit-module-maps \
// RUN:   -fmodules-cache-path=%t/cache -I %t -Rmodule-include-translation \
// RUN:   -verify %t/use.c

//--- module.modulemap
requires cplusplus {
  module cxxonly { header "cxxonly.h" }
}

//--- cxxonly.h
#define CXXONLY_MACRO 42

//--- use.c
// expected-no-diagnostics
#include "cxxonly.h"
int x = CXXONLY_MACRO;
