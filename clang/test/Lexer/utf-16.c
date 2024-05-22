// RUN: %clang_cc1 %s -fsyntax-only -verify

// This test verifies that clang gives a decent error for UTF-16 source files.

#include "utf-16.c.txt" // expected-error {{UTF-16 (LE) byte order mark detected}}
