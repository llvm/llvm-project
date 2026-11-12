// Test this without pch.
// RUN: %clang_cc1 -Wunused-macros -Dunused=1 -fsyntax-only -verify %s

// Test with pch.
// RUN: %clang_cc1 -Wunused-macros -emit-pch -o %t %s
// RUN: %clang_cc1 -Wunused-macros -Dunused=1 -include-pch %t -fsyntax-only -verify %s

// expected-no-diagnostics

// -Dunused=1 is intentionally not set for the pch.
// There still should be no unused warning for a macro from the command line.

