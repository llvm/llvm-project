// RUN: %clang_cc1 -ffreestanding -Eonly -verify %s

// Try different path permutations of __has_include with existing file.
#if __has_include("stdint.h")
#else
  #error "__has_include failed (1)."
#endif

#if __has_include(<stdint.h>)
#else
  #error "__has_include failed (2)."
#endif

// Try unary expression.
#if !__has_include("stdint.h")
  #error "__has_include failed (5)."
#endif

// Try binary expression.
#if __has_include("stdint.h") && __has_include("stddef.h")
#else
  #error "__has_include failed (6)."
#endif

// Try non-existing file.
#if __has_include("blahblah.h")
  #error "__has_include failed (7)."
#endif

// Try defined.
#if !defined(__has_include)
  #error "defined(__has_include) failed (8)."
#endif

// Try different path permutations of __has_include_next with existing file.
#if __has_include_next("stddef.h") // expected-warning {{#include_next in primary source file}}
#else
  #error "__has_include failed (1)."
#endif

#if __has_include_next(<stddef.h>) // expected-warning {{#include_next in primary source file}}
#else
  #error "__has_include failed (2)."
#endif

// Try unary expression.
#if !__has_include_next("stdint.h") // expected-warning {{#include_next in primary source file}}
  #error "__has_include_next failed (5)."
#endif

// Try binary expression.
#if __has_include_next("stdint.h") && __has_include("stddef.h") // expected-warning {{#include_next in primary source file}}
#else
  #error "__has_include_next failed (6)."
#endif

// Try non-existing file.
#if __has_include_next("blahblah.h") // expected-warning {{#include_next in primary source file}}
  #error "__has_include_next failed (7)."
#endif

// Try defined.
#if !defined(__has_include_next)
  #error "defined(__has_include_next) failed (8)."
#endif

// Try badly formed expressions.
// FIXME: We can recover better in almost all of these cases. (PR13335)

// expected-error@+1 {{missing '(' after '__has_include'}}
#if __has_include "stdint.h")
#endif

// expected-error@+1 {{expected "FILENAME" or <FILENAME>}} expected-error@+1 {{token is not a valid binary operator in a preprocessor subexpression}}
#if __has_include(stdint.h)
#endif

// expected-error@+1 {{expected "FILENAME" or <FILENAME>}}
#if __has_include()
#endif

// expected-error@+1 {{missing '(' after '__has_include'}}
#if __has_include)
#endif

// expected-error@+1 {{missing '(' after '__has_include'}}
#if __has_include<stdint.h>)
#endif

// expected-error@+1 {{expected "FILENAME" or <FILENAME>}} expected-warning@+1 {{missing terminating '"' character}}  expected-error@+1 {{invalid token at start of a preprocessor expression}}
#if __has_include("stdint.h)
#endif

// expected-error@+1 {{expected "FILENAME" or <FILENAME>}} expected-warning@+1 {{missing terminating '"' character}} expected-error@+1 {{token is not a valid binary operator in a preprocessor subexpression}}
#if __has_include(stdint.h")
#endif

// expected-error@+1 {{expected "FILENAME" or <FILENAME>}} expected-error@+1 {{token is not a valid binary operator in a preprocessor subexpression}}
#if __has_include(stdint.h>)
#endif

// expected-error@+1 {{missing '(' after '__has_include'}}
__has_include

// expected-error@+1 {{missing ')' after '__has_include'}} // expected-error@+1 {{expected value in expression}}  // expected-note@+1 {{to match this '('}}
#if __has_include("stdint.h"
#endif

// expected-error@+1 {{expected "FILENAME" or <FILENAME>}} // expected-error@+1 {{expected value in expression}}
#if __has_include(
#endif

// expected-error@+1 {{missing '(' after '__has_include'}} // expected-error@+1 {{expected value in expression}}
#if __has_include
#endif

// expected-error@+1 {{missing ')' after '__has_include'}}  // expected-error@+1 {{expected value in expression}}  // expected-note@+1 {{to match this '('}}
#if __has_include(<stdint.h>
#endif

// expected-error@+1 {{expected "FILENAME" or <FILENAME>}} // expected-error@+1 {{expected value in expression}}
#if __has_include(<stdint.h)
#endif
