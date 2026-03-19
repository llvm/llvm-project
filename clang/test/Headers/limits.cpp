// RUN: %clang_cc1 -ffreestanding -fsyntax-only -verify %s
// RUN: %clang_cc1 -fno-signed-char -ffreestanding -fsyntax-only -verify %s
// RUN: %clang_cc1 -std=c++11 -ffreestanding -fsyntax-only -verify %s
// RUN: %clang_cc1 -std=c17 -ffreestanding -fsyntax-only -verify -x c %s
// RUN: %clang_cc1 -std=c2x -ffreestanding -fsyntax-only -verify -x c %s

// Specifically test arm64 linux platforms.
// RUN: %clang_cc1 -triple arm64-linux -ffreestanding -fsyntax-only -verify -x c %s

// Specifically test 16-bit int platforms.
// RUN: %clang_cc1 -triple=avr -ffreestanding -fsyntax-only -verify -x c %s
// RUN: %clang_cc1 -triple=avr -std=c++11 -ffreestanding -fsyntax-only -verify %s

// expected-no-diagnostics

#include <limits.h>

#if __cplusplus
#define EXPR_TYPE_IS(EXPR, TYP) __is_same(__typeof(EXPR), TYP)
#else
#define EXPR_TYPE_IS(EXPR, TYP) _Generic(EXPR, TYP: 1, default: 0)
#endif

_Static_assert(SCHAR_MAX == -(SCHAR_MIN+1), "");
_Static_assert(EXPR_TYPE_IS(SCHAR_MAX, int), "");
#if SCHAR_MAX
#endif

_Static_assert(SHRT_MAX == -(SHRT_MIN+1), "");
_Static_assert(EXPR_TYPE_IS(SHRT_MAX, int), "");
#if SHRT_MAX
#endif

_Static_assert(INT_MAX == -(INT_MIN+1), "");
_Static_assert(EXPR_TYPE_IS(INT_MAX, int), "");
#if INT_MAX
#endif

_Static_assert(LONG_MAX == -(LONG_MIN+1L), "");
_Static_assert(EXPR_TYPE_IS(LONG_MAX, long), "");
#if LONG_MAX
#endif

_Static_assert(SCHAR_MAX == UCHAR_MAX/2, "");
_Static_assert(SHRT_MAX == USHRT_MAX/2, "");
_Static_assert(INT_MAX == UINT_MAX/2, "");
_Static_assert(LONG_MAX == ULONG_MAX/2, "");

_Static_assert(SCHAR_MIN == -SCHAR_MAX-1, "");
_Static_assert(EXPR_TYPE_IS(SCHAR_MIN, int), "");
#if SCHAR_MIN
#endif

_Static_assert(SHRT_MIN == -SHRT_MAX-1, "");
_Static_assert(EXPR_TYPE_IS(SHRT_MIN, int), "");
#if SHRT_MIN
#endif

_Static_assert(INT_MIN == -INT_MAX-1, "");
_Static_assert(EXPR_TYPE_IS(INT_MIN, int), "");
#if INT_MIN
#endif

_Static_assert(LONG_MIN == -LONG_MAX-1L, "");
_Static_assert(EXPR_TYPE_IS(LONG_MIN, long), "");
#if LONG_MIN
#endif

_Static_assert(UCHAR_MAX == (unsigned char)~0ULL, "");
_Static_assert(UCHAR_MAX <= INT_MAX ?
                 EXPR_TYPE_IS(UCHAR_MAX, int) :
                 EXPR_TYPE_IS(UCHAR_MAX, unsigned int), "");
#if UCHAR_MAX
#endif

_Static_assert(USHRT_MAX == (unsigned short)~0ULL, "");
_Static_assert(USHRT_MAX <= INT_MAX ?
                 EXPR_TYPE_IS(USHRT_MAX, int) :
                 EXPR_TYPE_IS(USHRT_MAX, unsigned int), "");
#if USHRT_MAX
#endif

_Static_assert(UINT_MAX == (unsigned int)~0ULL, "");
_Static_assert(EXPR_TYPE_IS(UINT_MAX, unsigned int), "");
#if UINT_MAX
#endif

_Static_assert(ULONG_MAX == (unsigned long)~0ULL, "");
_Static_assert(EXPR_TYPE_IS(ULONG_MAX, unsigned long), "");
#if ULONG_MAX
#endif

_Static_assert(MB_LEN_MAX >= 1, "");
#if MB_LEN_MAX
#endif

_Static_assert(CHAR_BIT >= 8, "");
#if CHAR_BIT
#endif

_Static_assert(CHAR_MIN == (((char)-1 < (char)0) ? -CHAR_MAX-1 : 0), "");
_Static_assert(EXPR_TYPE_IS(CHAR_MIN, int), "");
#if CHAR_MIN
#endif

_Static_assert(CHAR_MAX == (((char)-1 < (char)0) ? -(CHAR_MIN+1) : (char)~0ULL), "");
_Static_assert(CHAR_MAX <= INT_MAX ?
                 EXPR_TYPE_IS(CHAR_MAX, int) :
                 EXPR_TYPE_IS(CHAR_MAX, unsigned int), "");
#if CHAR_MAX
#endif

#if __STDC_VERSION__ >= 199901 || __cplusplus >= 201103L
_Static_assert(LLONG_MAX == -(LLONG_MIN+1LL), "");
_Static_assert(EXPR_TYPE_IS(LLONG_MAX, long long), "");
#if LLONG_MAX
#endif

_Static_assert(LLONG_MIN == -LLONG_MAX-1LL, "");
#if LLONG_MIN
#endif
_Static_assert(EXPR_TYPE_IS(LLONG_MIN, long long), "");

_Static_assert(ULLONG_MAX == (unsigned long long)~0ULL, "");
_Static_assert(EXPR_TYPE_IS(ULLONG_MAX, unsigned long long), "");
#if ULLONG_MAX
#endif
#else
int LLONG_MIN, LLONG_MAX, ULLONG_MAX; // Not defined.
#endif

#if __STDC_VERSION__ >= 202311L
/* Validate the standard requirements. */
_Static_assert(BOOL_WIDTH >= 1);
#if BOOL_WIDTH
#endif

_Static_assert(CHAR_WIDTH == CHAR_BIT);
_Static_assert(CHAR_WIDTH / CHAR_BIT == sizeof(char));
#if CHAR_WIDTH
#endif
_Static_assert(SCHAR_WIDTH == CHAR_BIT);
_Static_assert(SCHAR_WIDTH / CHAR_BIT == sizeof(signed char));
#if SCHAR_WIDTH
#endif
_Static_assert(UCHAR_WIDTH == CHAR_BIT);
_Static_assert(UCHAR_WIDTH / CHAR_BIT == sizeof(unsigned char));
#if UCHAR_WIDTH
#endif

_Static_assert(USHRT_WIDTH >= 16);
_Static_assert(USHRT_WIDTH / CHAR_BIT == sizeof(unsigned short));
#if USHRT_WIDTH
#endif
_Static_assert(SHRT_WIDTH == USHRT_WIDTH);
_Static_assert(SHRT_WIDTH / CHAR_BIT == sizeof(signed short));
#if SHRT_WIDTH
#endif

_Static_assert(UINT_WIDTH >= 16);
_Static_assert(UINT_WIDTH / CHAR_BIT == sizeof(unsigned int));
#if UINT_WIDTH
#endif
_Static_assert(INT_WIDTH == UINT_WIDTH);
_Static_assert(INT_WIDTH / CHAR_BIT == sizeof(signed int));
#if INT_WIDTH
#endif

_Static_assert(ULONG_WIDTH >= 32);
_Static_assert(ULONG_WIDTH / CHAR_BIT == sizeof(unsigned long));
#if ULONG_WIDTH
#endif
_Static_assert(LONG_WIDTH == ULONG_WIDTH);
_Static_assert(LONG_WIDTH / CHAR_BIT == sizeof(signed long));
#if LONG_WIDTH
#endif

_Static_assert(ULLONG_WIDTH >= 64);
_Static_assert(ULLONG_WIDTH / CHAR_BIT == sizeof(unsigned long long));
#if ULLONG_WIDTH
#endif
_Static_assert(LLONG_WIDTH == ULLONG_WIDTH);
_Static_assert(LLONG_WIDTH / CHAR_BIT == sizeof(signed long long));
#if LLONG_WIDTH
#endif

_Static_assert(BITINT_MAXWIDTH >= ULLONG_WIDTH);
#if BITINT_MAXWIDTH
#endif
#else
/* None of these are defined. */
int BOOL_WIDTH, CHAR_WIDTH, SCHAR_WIDTH, UCHAR_WIDTH, USHRT_WIDTH, SHRT_WIDTH,
    UINT_WIDTH, INT_WIDTH, ULONG_WIDTH, LONG_WIDTH, ULLONG_WIDTH, LLONG_WIDTH,
    BITINT_MAXWIDTH;
#endif
