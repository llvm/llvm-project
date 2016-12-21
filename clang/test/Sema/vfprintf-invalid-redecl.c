// RUN: %clang_cc1 %s -fsyntax-only -verify
// PR4290

// The following declaration is not compatible with vfprintf(), but make
// sure this isn't an error: autoconf expects this to build.
char vfprintf(); // expected-warning {{declaration of built-in function 'vfprintf'}}
