// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: touch %t/simple.h
// RUN: %clang_cc1 -fsyntax-only -I%t -Wnonportable-include-path -verify %s
// REQUIRES: system-windows

// On Windows, the filesystem silently strips trailing whitespace and dots
// from filenames, so the include succeeds. We should still emit a
// portability warning but no file-not-found error.

// Trailing whitespace: warn about non-portable path, but file is found.
#include "simple.h " // expected-warning {{non-portable path to file 'simple.h '; specified path contains trailing whitespace}}

// Trailing dots: warn about non-portable path, but file is found.
#include "simple.h." // expected-warning {{non-portable path to file 'simple.h.'; specified path contains trailing dot}}

// Correct path: no diagnostics expected.
#include "simple.h" // no-warning
