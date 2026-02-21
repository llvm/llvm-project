// Test that diagnostics are appropriately suppressed for included header
// files that match an external include directory prefix. These tests
// validate external path directory prefix matching in the presence of
// mismatched duplicate path separators and "." and ".." path components.
// Paths with partial path component matching are used as external include
// directory prefixes so that they do not nominate additional include paths.

// RUN: rm -rf %t
// RUN: split-file %s %t

// Test 1: Validate path matching for mismatched presence of "." path
// components.
//
// RUN: %clang_cc1 \
// RUN:     -fsyntax-only -verify -Wall -Wno-system-headers \
// RUN:     -I%t/inc/./foo \
// RUN:     -iexternal %t/inc/fo \
// RUN:     %t/should-not-warn.c
// RUN: %clang_cc1 \
// RUN:     -fsyntax-only -verify -Wall -Wno-system-headers \
// RUN:     -I%t/inc/foo \
// RUN:     -iexternal %t/inc/./fo \
// RUN:     %t/should-not-warn.c

// Test 2: Validate path matching for mismatched presence of ".." path
// components. Note that path normalization of an include path will not
// remove ".." path components for a non-existent parent path.
//
// RUN: %clang_cc1 \
// RUN:     -fsyntax-only -verify -Wall -Wno-system-headers \
// RUN:     -I%t/inc/foo \
// RUN:     -iexternal %t/inc/no-such-dir/../fo \
// RUN:     %t/should-not-warn.c

// Test 3: Validate path matching for mismatched duplication of path
// separators.
//
// RUN: %clang_cc1 \
// RUN:     -fsyntax-only -verify -Wall -Wno-system-headers \
// RUN:     -I%t/inc//foo \
// RUN:     -iexternal %t/inc/fo \
// RUN:     %t/should-not-warn.c
// RUN: %clang_cc1 \
// RUN:     -fsyntax-only -verify -Wall -Wno-system-headers \
// RUN:     -I%t/inc/foo \
// RUN:     -iexternal %t/inc//fo \
// RUN:     %t/should-not-warn.c

// Test 4: Validate path matching for mismatched relative and
// absolute paths.
//
// RUN: pushd %t
// RUN: %clang_cc1 \
// RUN:     -fsyntax-only -verify -Wall -Wno-system-headers \
// RUN:     -Iinc/foo \
// RUN:     -iexternal %t/inc/fo \
// RUN:     %t/should-not-warn.c
// RUN: %clang_cc1 \
// RUN:     -fsyntax-only -verify -Wall -Wno-system-headers \
// RUN:     -I%t/inc/foo \
// RUN:     -iexternal inc/fo \
// RUN:     %t/should-not-warn.c
// RUN: popd

#--- should-not-warn.c
#include <bar/should-not-warn.h>
// Validate that a warning is actually issued for test code intended to
// solicit a warning and that suppression of such a warning in an included
// header file does not affect warnings issued for other files.
// expected-warning@+1 {{shift count >= width of type}}
int x = 1 << 1024; // Warning should not be suppressed.

#--- inc/foo/bar/should-not-warn.h
int h = 1 << 1024; // Warning should be suppressed.
