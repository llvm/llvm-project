// REQUIRES: system-darwin
// RUN: %clang -target x86_64-apple-darwin11 -fsyntax-only -std=c11 -isysroot %S/Inputs %s
#include <float.h>

// Test the #include_next on float.h works on Darwin.
#ifndef FLT_HAS_SUBNORM
  #error "FLT_HAS_SUBNORM not defined"
#endif

// Test that definition from builtin are also present.
#ifndef FLT_MAX
  #error "FLT_MAX not defined"
#endif
