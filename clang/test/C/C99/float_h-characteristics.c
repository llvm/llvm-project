// RUN: %clang_cc1 -verify -std=c99 -ffreestanding %s
// RUN: %clang_cc1 -verify -std=gnu89 -ffreestanding %s
// RUN: %clang_cc1 -verify -std=c89 -ffreestanding %s
// expected-no-diagnostics

/* WG14 ???: Clang 16
 * Additional floating-point characteristics in <float.h>
 *
 * NB: the original paper number is unknown, this was gleaned from the editor's
 * report in the C99 foreword. There were two new additions to <float.h> in
 * C99, this is testing that we support both of them.
 *
 * Clang added the macros at least as far back as Clang 3.0, but it wasn't
 * until Clang 16.0 that we stopped accidentally providing FLT_EVAL_METHOD in
 * C89 (strict) mode.
 */

#include <float.h>

// We expect all the definitions in C99 mode.
#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 199901L
#define EXPECT_DECIMAL_DIG
#define EXPECT_FLT_EVAL_METHOD
#endif

// If we're not in C99 mode, we still expect the definition of DECIMAL_DIG
// unless we're in strict ansi mode.
#if !defined(EXPECT_DECIMAL_DIG) && !defined(__STRICT_ANSI__)
#define EXPECT_DECIMAL_DIG
#endif

#if defined(EXPECT_DECIMAL_DIG)
  #if !defined(DECIMAL_DIG)
    #error "DECIMAL_DIG missing"
  #endif
#else
  #if defined(DECIMAL_DIG)
    #error "DECIMAL_DIG provided when not expected"
  #endif
#endif

#if defined(EXPECT_FLT_EVAL_METHOD)
  #if !defined(FLT_EVAL_METHOD)
    #error "FLT_EVAL_METHOD missing"
  #endif
#else
  #if defined(FLT_EVAL_METHOD)
    #error "FLT_EVAL_METHOD provided when not expected"
  #endif
#endif
