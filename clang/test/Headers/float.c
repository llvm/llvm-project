// RUN: %clang_cc1 -fsyntax-only -verify -std=c89 -ffreestanding %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c99 -ffreestanding %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c11 -ffreestanding %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c23 -ffreestanding %s
// RUN: %clang_cc1 -fsyntax-only -verify=finite -std=c23 -ffreestanding -menable-no-nans -menable-no-infs %s
// RUN: %clang_cc1 -fsyntax-only -verify -xc++ -std=c++11 -ffreestanding %s
// RUN: %clang_cc1 -fsyntax-only -verify -xc++ -std=c++14 -ffreestanding %s
// RUN: %clang_cc1 -fsyntax-only -verify -xc++ -std=c++17 -ffreestanding %s
// RUN: %clang_cc1 -fsyntax-only -verify -xc++ -std=c++23 -ffreestanding %s
// NOTE: C++23 wasn't based on top of C23, so it gets no diagnostics for
//       finite-math-only mode as happens in C. When C++ rebased onto C23, that
//       is when we'll issue diagnostics for INFINITY and NAN use.
// RUN: %clang_cc1 -fsyntax-only -verify -xc++ -std=c++23 -ffreestanding -ffinite-math-only %s
// expected-no-diagnostics

/* Basic floating point conformance checks against:
    - C23 Final Std.
    - N1570 draft of C11 Std.
    - N1256 draft of C99 Std.
    - http://port70.net/~nsz/c/c89/c89-draft.html draft of C89/C90 Std.
*/
/*
    C23,    5.2.5.3.3p21,   pp. 25
    C11,    5.2.4.2.2p11,   pp. 30
    C99,    5.2.4.2.2p9,    pp. 25
    C89,    2.2.4.2
*/
#include <float.h>

#ifndef FLT_RADIX
    #error "Mandatory macro FLT_RADIX is missing."
#elif   FLT_RADIX < 2
    #error "Mandatory macro FLT_RADIX is invalid."
#endif


#ifndef FLT_MANT_DIG
    #error "Mandatory macro FLT_MANT_DIG is missing."
#elif   FLT_MANT_DIG < 2
    #error "Mandatory macro FLT_MANT_DIG is invalid."
#endif
#ifndef DBL_MANT_DIG
    #error "Mandatory macro DBL_MANT_DIG is missing."
#elif   DBL_MANT_DIG < 2
    #error "Mandatory macro DBL_MANT_DIG is invalid."
#endif
#ifndef LDBL_MANT_DIG
    #error "Mandatory macro LDBL_MANT_DIG is missing."
#elif   LDBL_MANT_DIG < 2
    #error "Mandatory macro LDBL_MANT_DIG is invalid."
#endif
#if ((FLT_MANT_DIG > DBL_MANT_DIG) || (DBL_MANT_DIG > LDBL_MANT_DIG))
    #error "Mandatory macros {FLT,DBL,LDBL}_MANT_DIG are invalid."
#endif


#if __STDC_VERSION__ >= 201112L || !defined(__STRICT_ANSI__) || __cplusplus >= 201703L
    #ifndef FLT_DECIMAL_DIG
        #error "Mandatory macro FLT_DECIMAL_DIG is missing."
    #elif   FLT_DECIMAL_DIG < 6
        #error "Mandatory macro FLT_DECIMAL_DIG is invalid."
    #endif
    #ifndef DBL_DECIMAL_DIG
        #error "Mandatory macro DBL_DECIMAL_DIG is missing."
    #elif   DBL_DECIMAL_DIG < 10
        #error "Mandatory macro DBL_DECIMAL_DIG is invalid."
    #endif
    #ifndef LDBL_DECIMAL_DIG
        #error "Mandatory macro LDBL_DECIMAL_DIG is missing."
    #elif   LDBL_DECIMAL_DIG < 10
        #error "Mandatory macro LDBL_DECIMAL_DIG is invalid."
    #endif
    #if ((FLT_DECIMAL_DIG > DBL_DECIMAL_DIG) || (DBL_DECIMAL_DIG > LDBL_DECIMAL_DIG))
        #error "Mandatory macros {FLT,DBL,LDBL}_DECIMAL_DIG are invalid."
    #endif
    #ifndef FLT_HAS_SUBNORM
        #error "Mandatory macro FLT_HAS_SUBNORM is missing."
    #elif FLT_HAS_SUBNORM != __FLT_HAS_DENORM__
        #error "Mandatory macro FLT_HAS_SUBNORM is invalid."
    #endif
    #ifndef LDBL_HAS_SUBNORM
        #error "Mandatory macro LDBL_HAS_SUBNORM is missing."
    #elif LDBL_HAS_SUBNORM != __LDBL_HAS_DENORM__
        #error "Mandatory macro LDBL_HAS_SUBNORM is invalid."
    #endif
    #ifndef DBL_HAS_SUBNORM
        #error "Mandatory macro DBL_HAS_SUBNORM is missing."
    #elif DBL_HAS_SUBNORM != __DBL_HAS_DENORM__
        #error "Mandatory macro DBL_HAS_SUBNORM is invalid."
    #endif
#else
    #ifdef FLT_DECIMAL_DIG
        #error "Macro FLT_DECIMAL_DIG should not be defined."
    #endif
    #ifdef DBL_DECIMAL_DIG
        #error "Macro DBL_DECIMAL_DIG should not be defined."
    #endif
    #ifdef LDBL_DECIMAL_DIG
        #error "Macro LDBL_DECIMAL_DIG should not be defined."
    #endif
    #ifdef FLT_HAS_SUBNORM
        #error "Macro FLT_HAS_SUBNORM should not be defined."
    #endif
    #ifdef DBL_HAS_SUBNORM
        #error "Macro DBL_HAS_SUBNORM should not be defined."
    #endif
    #ifdef LDBL_HAS_SUBNORM
        #error "Macro LDBL_HAS_SUBNORM should not be defined."
    #endif
#endif


#if __STDC_VERSION__ >= 199901L || !defined(__STRICT_ANSI__) || __cplusplus >= 201103L
    #ifndef DECIMAL_DIG
        #error "Mandatory macro DECIMAL_DIG is missing."
    #elif   DECIMAL_DIG < 10
        #error "Mandatory macro DECIMAL_DIG is invalid."
    #endif
#else
    #ifdef DECIMAL_DIG
        #error "Macro DECIMAL_DIG should not be defined."
    #endif
#endif


#ifndef FLT_DIG
    #error "Mandatory macro FLT_DIG is missing."
#elif   FLT_DIG < 6
    #error "Mandatory macro FLT_DIG is invalid."
#endif
#ifndef DBL_DIG
    #error "Mandatory macro DBL_DIG is missing."
#elif   DBL_DIG < 10
    #error "Mandatory macro DBL_DIG is invalid."
#endif
#ifndef LDBL_DIG
    #error "Mandatory macro LDBL_DIG is missing."
#elif   LDBL_DIG < 10
    #error "Mandatory macro LDBL_DIG is invalid."
#endif
#if ((FLT_DIG > DBL_DIG) || (DBL_DIG > LDBL_DIG))
    #error "Mandatory macros {FLT,DBL,LDBL}_DIG, are invalid."
#endif


#ifndef FLT_MIN_EXP
    #error "Mandatory macro FLT_MIN_EXP is missing."
#elif   FLT_MIN_EXP > -1
    #error "Mandatory macro FLT_MIN_EXP is invalid."
#endif
#ifndef DBL_MIN_EXP
    #error "Mandatory macro DBL_MIN_EXP is missing."
#elif   DBL_MIN_EXP > -1
    #error "Mandatory macro DBL_MIN_EXP is invalid."
#endif
#ifndef LDBL_MIN_EXP
    #error "Mandatory macro LDBL_MIN_EXP is missing."
#elif   LDBL_MIN_EXP > -1
    #error "Mandatory macro LDBL_MIN_EXP is invalid."
#endif


#ifndef FLT_MIN_10_EXP
    #error "Mandatory macro FLT_MIN_10_EXP is missing."
#elif   FLT_MIN_10_EXP > -37
    #error "Mandatory macro FLT_MIN_10_EXP is invalid."
#endif
#ifndef DBL_MIN_10_EXP
    #error "Mandatory macro DBL_MIN_10_EXP is missing."
#elif   DBL_MIN_10_EXP > -37
    #error "Mandatory macro DBL_MIN_10_EXP is invalid."
#endif
#ifndef LDBL_MIN_10_EXP
    #error "Mandatory macro LDBL_MIN_10_EXP is missing."
#elif   LDBL_MIN_10_EXP > -37
    #error "Mandatory macro LDBL_MIN_10_EXP is invalid."
#endif


#ifndef FLT_MAX_EXP
    #error "Mandatory macro FLT_MAX_EXP is missing."
#elif   FLT_MAX_EXP < 1
    #error "Mandatory macro FLT_MAX_EXP is invalid."
#endif
#ifndef DBL_MAX_EXP
    #error "Mandatory macro DBL_MAX_EXP is missing."
#elif   DBL_MAX_EXP < 1
    #error "Mandatory macro DBL_MAX_EXP is invalid."
#endif
#ifndef LDBL_MAX_EXP
    #error "Mandatory macro LDBL_MAX_EXP is missing."
#elif   LDBL_MAX_EXP < 1
    #error "Mandatory macro LDBL_MAX_EXP is invalid."
#endif
#if ((FLT_MAX_EXP > DBL_MAX_EXP) || (DBL_MAX_EXP > LDBL_MAX_EXP))
    #error "Mandatory macros {FLT,DBL,LDBL}_MAX_EXP are invalid."
#endif


#ifndef FLT_MAX_10_EXP
    #error "Mandatory macro FLT_MAX_10_EXP is missing."
#elif   FLT_MAX_10_EXP < 37
    #error "Mandatory macro FLT_MAX_10_EXP is invalid."
#endif
#ifndef DBL_MAX_10_EXP
    #error "Mandatory macro DBL_MAX_10_EXP is missing."
#elif   DBL_MAX_10_EXP < 37
    #error "Mandatory macro DBL_MAX_10_EXP is invalid."
#endif
#ifndef LDBL_MAX_10_EXP
    #error "Mandatory macro LDBL_MAX_10_EXP is missing."
#elif   LDBL_MAX_10_EXP < 37
    #error "Mandatory macro LDBL_MAX_10_EXP is invalid."
#endif
#if ((FLT_MAX_10_EXP > DBL_MAX_10_EXP) || (DBL_MAX_10_EXP > LDBL_MAX_10_EXP))
    #error "Mandatory macros {FLT,DBL,LDBL}_MAX_10_EXP are invalid."
#endif

#if __STDC_VERSION__ >= 202311L || !defined(__STRICT_ANSI__)
  #ifndef INFINITY
    #error "Mandatory macro INFINITY is missing."
  #endif
  #ifndef NAN
    #error "Mandatory macro NAN is missing."
  #endif
// FIXME: the NAN and INF diagnostics should only be issued once, not twice.
  _Static_assert(_Generic(INFINITY, float : 1, default : 0), ""); // finite-warning {{use of infinity via a macro is undefined behavior due to the currently enabled floating-point options}} \
								  finite-warning {{use of infinity is undefined behavior due to the currently enabled floating-point options}}
  _Static_assert(_Generic(NAN, float : 1, default : 0), ""); // finite-warning {{use of NaN is undefined behavior due to the currently enabled floating-point options}} \
                                                                finite-warning {{use of NaN via a macro is undefined behavior due to the currently enabled floating-point options}}

#ifndef FLT_NORM_MAX
  #error "Mandatory macro FLT_NORM_MAX is missing."
#else
  _Static_assert(FLT_NORM_MAX >= 1.0E+37F, "Mandatory macro FLT_NORM_MAX is invalid.");
#endif
#ifndef DBL_NORM_MAX
  #error "Mandatory macro DBL_NORM_MAX is missing."
#else
  _Static_assert(DBL_NORM_MAX >= 1.0E+37, "Mandatory macro DBL_NORM_MAX is invalid.");
#endif
#ifndef LDBL_NORM_MAX
  #error "Mandatory macro LDBL_NORM_MAX is missing."
#else
  _Static_assert(LDBL_NORM_MAX >= 1.0E+37L, "Mandatory macro LDBL_NORM_MAX is invalid.");
#endif
#else
  #ifdef INFINITY
    #error "Macro INFINITY should not be defined."
  #endif
  #ifdef NAN
    #error "Macro NAN should not be defined."
  #endif
#endif

/* Internal consistency checks */
_Static_assert(FLT_RADIX == __FLT_RADIX__, "");

_Static_assert(FLT_MANT_DIG == __FLT_MANT_DIG__, "");
_Static_assert(DBL_MANT_DIG == __DBL_MANT_DIG__, "");
_Static_assert(LDBL_MANT_DIG == __LDBL_MANT_DIG__, "");

#if __STDC_VERSION__ >= 201112L || !defined(__STRICT_ANSI__) || __cplusplus >= 201703L
_Static_assert(FLT_DECIMAL_DIG == __FLT_DECIMAL_DIG__, "");
_Static_assert(DBL_DECIMAL_DIG == __DBL_DECIMAL_DIG__, "");
_Static_assert(LDBL_DECIMAL_DIG == __LDBL_DECIMAL_DIG__, "");
#endif

#if __STDC_VERSION__ >= 199901L || !defined(__STRICT_ANSI__) || __cplusplus >= 201103L
_Static_assert(DECIMAL_DIG == __DECIMAL_DIG__, "");
#endif

_Static_assert(FLT_DIG == __FLT_DIG__, "");
_Static_assert(DBL_DIG == __DBL_DIG__, "");
_Static_assert(LDBL_DIG == __LDBL_DIG__, "");

_Static_assert(FLT_MIN_EXP == __FLT_MIN_EXP__, "");
_Static_assert(DBL_MIN_EXP == __DBL_MIN_EXP__, "");
_Static_assert(LDBL_MIN_EXP == __LDBL_MIN_EXP__, "");

_Static_assert(FLT_MIN_10_EXP == __FLT_MIN_10_EXP__, "");
_Static_assert(DBL_MIN_10_EXP == __DBL_MIN_10_EXP__, "");
_Static_assert(LDBL_MIN_10_EXP == __LDBL_MIN_10_EXP__, "");

_Static_assert(FLT_MAX_EXP == __FLT_MAX_EXP__, "");
_Static_assert(DBL_MAX_EXP == __DBL_MAX_EXP__, "");
_Static_assert(LDBL_MAX_EXP == __LDBL_MAX_EXP__, "");

_Static_assert(FLT_MAX_10_EXP == __FLT_MAX_10_EXP__, "");
_Static_assert(DBL_MAX_10_EXP == __DBL_MAX_10_EXP__, "");
_Static_assert(LDBL_MAX_10_EXP == __LDBL_MAX_10_EXP__, "");

#if __STDC_VERSION__ >= 202311L || !defined(__STRICT_ANSI__)
_Static_assert(FLT_NORM_MAX == __FLT_NORM_MAX__, "");
_Static_assert(DBL_NORM_MAX == __DBL_NORM_MAX__, "");
_Static_assert(LDBL_NORM_MAX == __LDBL_NORM_MAX__, "");

#if __FINITE_MATH_ONLY__ == 0
// Ensure INFINITY and NAN are suitable for use in a constant expression.
float f1 = INFINITY;
float f2 = NAN;
#endif
#endif
