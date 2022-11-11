// REQUIRES: system-darwin
// RUN: %clang -target x86_64-apple-darwin11 -fsyntax-only -std=c11 -isysroot %S/Inputs %s
#include <tgmath.h>

// Test the #include_next of tgmath.h works on Darwin.
#ifndef SYS_TGMATH_H
  #error "SYS_TGMATH_H not defined"
#endif

#ifndef __CLANG_TGMATH_H
  #error "__CLANG_TGMATH_H not defined"
#endif
