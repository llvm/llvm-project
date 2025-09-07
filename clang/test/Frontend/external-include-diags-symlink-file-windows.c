// Test that diagnostics are appropriately suppressed for included header
// files that match an external include directory prefix. These tests
// validate external path directory prefix matching in the presence of
// file symbolic links. Paths with partial path component matching are
// used as external include directory prefixes so that they do not nominate
// additional include paths.

// REQUIRES: system-windows
// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: cmd /c mklink %t\inc1\foo\bar\should-warn.h ..\..\..\inc2\foo\bar\should-warn.h
// RUN: cmd /c mklink %t\inc1\foo\bar\should-not-warn.h ..\..\..\inc2\foo\bar\should-not-warn.h

// Test 1: Validate path matching with consistent redirection through a
// symbolic link.
//
// RUN: %clang_cc1 \
// RUN:     -fsyntax-only -verify -Wall -Wno-system-headers \
// RUN:     -I%t\inc1\foo \
// RUN:     -iexternal %t\inc1\foo\bar\sh \
// RUN:     %t\should-not-warn.c

// Test 2: Validate path matching with mismatched redirection through a
// symbolic link.
//
// RUN: %clang_cc1 \
// RUN:     -fsyntax-only -verify -Wall -Wno-system-headers \
// RUN:     -I%t\inc1\foo \
// RUN:     -iexternal %t\inc2\fo \
// RUN:     %t\should-warn.c
// RUN: %clang_cc1 \
// RUN:     -fsyntax-only -verify -Wall -Wno-system-headers \
// RUN:     -I%t\inc2\foo \
// RUN:     -iexternal %t\inc1\fo \
// RUN:     %t\should-warn.c

#--- should-not-warn.c
#include <bar/should-not-warn.h>
// Validate that a warning is actually issued for test code intended to
// solicit a warning and that suppression of such a warning in an included
// header file does not affect warnings issued for other files.
// expected-warning@+1 {{shift count >= width of type}}
int x = 1 << 1024; // Warning should not be suppressed.

#--- should-warn.c
#include <bar/should-warn.h>
// expected-warning@+1 {{shift count >= width of type}}
int x = 1 << 1024; // Warning should not be suppressed.

#--- inc1/foo/bar/unused.txt
Unused file used to ensure directory creation.

#--- inc2/foo/bar/should-not-warn.h
int h = 1 << 1024; // Warning should be suppressed.

#--- inc2/foo/bar/should-warn.h
// expected-warning@+1 {{shift count >= width of type}}
int h = 1 << 1024; // Warning should not be suppressed.
