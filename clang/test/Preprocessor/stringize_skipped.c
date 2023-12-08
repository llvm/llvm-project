// RUN: %clang_cc1 -fsyntax-only -verify %s
// Ensure we see the error from PP and do not see errors from the parser.

// expected-error@+1{{'#' is not followed by a macro parameter}}
#define INVALID() #B 10+10
