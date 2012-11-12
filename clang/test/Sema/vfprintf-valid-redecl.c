// RUN: %clang_cc1 %s -fsyntax-only -pedantic -verify
// expected-no-diagnostics
// PR4290

// The following declaration is compatible with vfprintf, so we shouldn't
// warn.
int vfprintf();
