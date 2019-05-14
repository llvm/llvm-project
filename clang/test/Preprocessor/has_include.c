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

// Fun with macros
#define MACRO1 __has_include(<stdint.h>)
#define MACRO2 ("stdint.h")
#define MACRO3 ("blahblah.h")
#define MACRO4 blahblah.h>)
#define MACRO5 <stdint.h>

#if !MACRO1
  #error "__has_include with macro failed (1)."
#endif

#if !__has_include MACRO2
  #error "__has_include with macro failed (2)."
#endif

#if __has_include MACRO3
  #error "__has_include with macro failed (3)."
#endif

#if __has_include(<MACRO4
  #error "__has_include with macro failed (4)."
#endif

#if !__has_include(MACRO5)
  #error "__has_include with macro failed (2)."
#endif

// Try as non-preprocessor directives
void foo( void ) {
  __has_include_next("stdint.h")  // expected-warning {{#include_next in primary source file}} expected-error {{'__has_include_next' must be used within a preprocessing directive}}
  __has_include("stdint.h")  // expected-error {{'__has_include' must be used within a preprocessing directive}}
}

MACRO1  // expected-error {{'__has_include' must be used within a preprocessing directive}}

#if 1
MACRO1  // expected-error {{'__has_include' must be used within a preprocessing directive}}
#endif

#if 0
#elif 1
MACRO1  // expected-error {{'__has_include' must be used within a preprocessing directive}}
#endif

#if 0
MACRO1  // This should be fine because it is never actually reached
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

// expected-error@+1 {{'__has_include' must be used within a preprocessing directive}}
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

// expected-error@+1 {{missing '(' after '__has_include'}}
#if __has_include'x'
#endif

// expected-error@+1 {{expected "FILENAME" or <FILENAME>}}
#if __has_include('x'
#endif

// expected-error@+1 {{expected "FILENAME" or <FILENAME}} expected-error@+1 {{expected end of line in preprocessor expression}}
#if __has_include('x')
#endif

// expected-error@+1 {{missing ')' after '__has_include'}}  // expected-error@+1 {{expected value in expression}}  // expected-note@+1 {{to match this '('}}
#if __has_include(<stdint.h>
#endif

// expected-error@+1 {{expected '>'}} expected-note@+1 {{to match this '<'}} // expected-error@+1 {{expected value in expression}}
#if __has_include(<stdint.h)
#endif

#define HAS_INCLUDE(header) __has_include(header)
#if HAS_INCLUDE(<stdint.h>)
#else
  #error "__has_include failed (9)."
#endif

#if FOO
#elif __has_include(<foo>)
#endif

// PR15539
#ifdef FOO
#elif __has_include(<foo>)
#endif
