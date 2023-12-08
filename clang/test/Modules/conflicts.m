// RUN: rm -rf %t
// RUN: %clang_cc1 -Rmodule-include-translation -fmodules-cache-path=%t -fmodules -fimplicit-module-maps -I %S/Inputs/Conflicts %s -verify

@import Conflicts;

@import Conflicts.A; // expected-warning{{module 'Conflicts.A' conflicts with already-imported module 'Conflicts.B': we just don't like B}}

