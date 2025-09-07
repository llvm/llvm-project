// Test that diagnostics are appropriately suppressed for included header
// files that match an external include directory prefix. These tests
// validate external path directory prefix matching in the presence of
// path components that contain space characters. Paths with partial path
// component matching are used as external include directory prefixes so
// that they do not nominate additional include paths.

// RUN: rm -rf %t
// RUN: split-file %s %t

// Test 1: Validate path matching for directories that contain spaces.
//
// RUN: %clang_cc1 \
// RUN:     -fsyntax-only -verify -Wall -Wno-system-headers \
// RUN:     -I"%t/inc/dir with spaces" \
// RUN:     -iexternal "%t/inc/dir with" \
// RUN:     %t/dir-with-spaces.c

// Test 2: Validate path matching for files that contain spaces.
//
// RUN: %clang_cc1 \
// RUN:     -fsyntax-only -verify -Wall -Wno-system-headers \
// RUN:     -I%t/inc/foo \
// RUN:     -iexternal "%t/inc/foo/bar/should not" \
// RUN:     %t/file-with-spaces.c

#--- dir-with-spaces.c
#include <bar/should-not-warn.h>
// Validate that a warning is actually issued for test code intended to
// solicit a warning and that suppression of such a warning in an included
// header file does not affect warnings issued for other files.
// expected-warning@+1 {{shift count >= width of type}}
int x = 1 << 1024; // Warning should not be suppressed.

#--- inc/dir with spaces/bar/should-not-warn.h
int h = 1 << 1024; // Warning should be suppressed.

#--- file-with-spaces.c
#include <bar/should not warn.h>
// Validate that a warning is actually issued for test code intended to
// solicit a warning and that suppression of such a warning in an included
// header file does not affect warnings issued for other files.
// expected-warning@+1 {{shift count >= width of type}}
int x = 1 << 1024; // Warning should not be suppressed.

#--- inc/foo/bar/should not warn.h
int h = 1 << 1024; // Warning should be suppressed.
