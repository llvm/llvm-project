// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: touch %t/simple.h
// RUN: %clang_cc1 -fsyntax-only -I%t -Wnonportable-include-path -verify %s
// UNSUPPORTED: system-windows

// On non-Windows systems, trailing whitespace and dots in include paths
// produce both a portability warning and a file-not-found error,
// because the filesystem treats them as part of the filename.

// Trailing whitespace: warn about non-portable path, error because file not found.
#include "simple.h " // expected-warning {{non-portable path to file 'simple.h '; specified path contains trailing whitespace}} \
                     // expected-error {{'simple.h ' file not found}}

// Trailing dots: warn about non-portable path, error because file not found.
#include "simple.h." // expected-warning {{non-portable path to file 'simple.h.'; specified path contains trailing dot}} \
                     // expected-error {{'simple.h.' file not found}}

// Correct path: no diagnostics expected.
#include "simple.h" // no-warning
