// Test that diagnostics are appropriately suppressed for included header
// files that match an external include directory prefix. These tests
// validate external path directory prefix matching in the presence of
// directory symbolic links. Paths with partial path component matching
// are used as external include directory prefixes so that they do not
// nominate additional include paths.

// REQUIRES: !system-windows
// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: ln -sf ../inc2/foo %t/inc1/foo
// RUN: ln -sf . %t/inc1/goo/baz

// Test 1: Validate path matching with consistent redirection through a
// symbolic link.
//
// RUN: %clang_cc1 \
// RUN:     -fsyntax-only -verify -Wall -Wno-system-headers \
// RUN:     -I%t/inc1/foo \
// RUN:     -iexternal %t/inc1/foo/b \
// RUN:     %t/should-not-warn.c

// Test 2: Validate path matching with mismatched redirection through a
// symbolic link.
//
// RUN: %clang_cc1 \
// RUN:     -fsyntax-only -verify -Wall -Wno-system-headers \
// RUN:     -I%t/inc1/foo \
// RUN:     -iexternal %t/inc2/fo \
// RUN:     %t/should-warn.c
// RUN: %clang_cc1 \
// RUN:     -triple x86_64-unknown-linux-gnu -fsyntax-only -verify \
// RUN:     -Wall -Wno-system-headers \
// RUN:     -I%t/inc2/foo \
// RUN:     -iexternal %t/inc1/fo \
// RUN:     %t/should-warn.c

// Test 3: Validate path matching with mismatched redirection through a
// symbolic link with path normalization removal of unnecessary ".." path
// components.
//
// RUN: %clang_cc1 \
// RUN:     -fsyntax-only -verify -Wall -Wno-system-headers \
// RUN:     -I%t/inc1/foo/../foo \
// RUN:     -iexternal %t/inc1/fo \
// RUN:     %t/should-not-warn.c
// RUN: %clang_cc1 \
// RUN:     -fsyntax-only -verify -Wall -Wno-system-headers \
// RUN:     -I%t/inc1/foo \
// RUN:     -iexternal %t/inc1/foo/../fo \
// RUN:     %t/should-not-warn.c

// Test 4: Validate path matching with mismatched redirection through a
// symbolic link for which removal of ".." path components is not possible.
//
// RUN: %clang_cc1 \
// RUN:     -fsyntax-only -verify -Wall -Wno-system-headers \
// RUN:     -I%t/inc1/goo/baz/../foo \
// RUN:     -iexternal %t/inc2/fo \
// RUN:     %t/should-warn.c
// RUN: %clang_cc1 \
// RUN:     -fsyntax-only -verify -Wall -Wno-system-headers \
// RUN:     -I%t/inc2/foo \
// RUN:     -iexternal %t/inc1/goo/baz/../fo \
// RUN:     %t/should-warn.c

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

#--- inc1/goo/unused.txt
Unused file used to ensure directory creation.

#--- inc2/foo/bar/should-not-warn.h
int h = 1 << 1024; // Warning should be suppressed.

#--- inc2/foo/bar/should-warn.h
// expected-warning@+1 {{shift count >= width of type}}
int h = 1 << 1024; // Warning should not be suppressed.
