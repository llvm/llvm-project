// RUN: %clang_cc1 -triple=aarch64-none-none -fsyntax-only -verify -std=c89 \
// RUN:   -ffreestanding %s
// RUN: %clang_cc1 -triple=aarch64-none-none -fsyntax-only -verify \
// RUN:   -std=c99 -ffreestanding %s
// RUN: %clang_cc1 -triple=aarch64-none-none -fsyntax-only -verify -std=c11 \
// RUN:   -ffreestanding %s
// RUN: %clang_cc1 -triple=aarch64-none-none -fsyntax-only -verify \
// RUN:   -std=c++11 -x c++ -ffreestanding %s
// expected-no-diagnostics

#define __STDC_WANT_IEC_60559_TYPES_EXT__
#include <float.h>

#ifndef FLT16_MIN_10_EXP
    #error "Macro FLT16_MIN_10_EXP is missing."
#elif   FLT16_MIN_10_EXP > -13
    #error "Macro FLT16_MIN_10_EXP is invalid."
#endif

_Static_assert(FLT16_MIN_10_EXP == __FLT16_MIN_10_EXP__, "");

#ifndef FLT16_MIN_EXP
    #error "Macro FLT16_MIN_EXP is missing."
#elif   FLT16_MIN_EXP > -14
    #error "Macro FLT16_MIN_EXP is invalid."
#endif

_Static_assert(FLT16_MIN_EXP == __FLT16_MIN_EXP__, "");

#ifndef FLT16_MAX_10_EXP
    #error "Macro FLT16_MAX_10_EXP is missing."
#elif   FLT16_MAX_10_EXP < 4
    #error "Macro FLT16_MAX_10_EXP is invalid."
#endif

_Static_assert(FLT16_MAX_10_EXP == __FLT16_MAX_10_EXP__, "");

#ifndef FLT16_MAX_EXP
    #error "Macro FLT16_MAX_EXP is missing."
#elif   FLT16_MAX_EXP < 15
    #error "Macro FLT16_MAX_EXP is invalid."
#endif

_Static_assert(FLT16_MAX_EXP == __FLT16_MAX_EXP__, "");

#ifndef FLT16_DECIMAL_DIG
    #error "Macro FLT16_DECIMAL_DIG is missing."
#elif   FLT16_DECIMAL_DIG < 5
    #error "Macro FLT16_DECIMAL_DIG is invalid."
#endif

_Static_assert(FLT16_DECIMAL_DIG == __FLT16_DECIMAL_DIG__, "");

#ifndef FLT16_DIG
    #error "Macro FLT16_DIG is missing."
#elif   FLT16_DIG < 3
    #error "Macro FLT16_DIG is invalid."
#endif

_Static_assert(FLT16_DIG == __FLT16_DIG__, "");

#ifndef FLT16_MANT_DIG
    #error "Macro FLT16_MANT_DIG is missing."
#elif   FLT16_MANT_DIG < 11
    #error "Macro FLT16_MANT_DIG is invalid."
#endif

_Static_assert(FLT16_MANT_DIG == __FLT16_MANT_DIG__, "");

