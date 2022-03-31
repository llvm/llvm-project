// RUN: %clang_cc1 -ffreestanding -fsyntax-only -verify %s
// RUN: %clang_cc1 -fno-signed-char -ffreestanding -fsyntax-only -verify %s
// RUN: %clang_cc1 -std=c++11 -ffreestanding -fsyntax-only -verify %s
// RUN: %clang_cc1 -std=c17 -ffreestanding -fsyntax-only -verify -x c %s
// RUN: %clang_cc1 -std=c2x -ffreestanding -fsyntax-only -verify -x c %s
// expected-no-diagnostics

#include <limits.h>

_Static_assert(SCHAR_MAX == -(SCHAR_MIN+1), "");
_Static_assert(SHRT_MAX == -(SHRT_MIN+1), "");
_Static_assert(INT_MAX == -(INT_MIN+1), "");
_Static_assert(LONG_MAX == -(LONG_MIN+1L), "");

_Static_assert(SCHAR_MAX == UCHAR_MAX/2, "");
_Static_assert(SHRT_MAX == USHRT_MAX/2, "");
_Static_assert(INT_MAX == UINT_MAX/2, "");
_Static_assert(LONG_MAX == ULONG_MAX/2, "");

_Static_assert(SCHAR_MIN == -SCHAR_MAX-1, "");
_Static_assert(SHRT_MIN == -SHRT_MAX-1, "");
_Static_assert(INT_MIN == -INT_MAX-1, "");
_Static_assert(LONG_MIN == -LONG_MAX-1L, "");

_Static_assert(UCHAR_MAX == (unsigned char)~0ULL, "");
_Static_assert(USHRT_MAX == (unsigned short)~0ULL, "");
_Static_assert(UINT_MAX == (unsigned int)~0ULL, "");
_Static_assert(ULONG_MAX == (unsigned long)~0ULL, "");

_Static_assert(MB_LEN_MAX >= 1, "");

_Static_assert(CHAR_BIT >= 8, "");

_Static_assert(CHAR_MIN == (((char)-1 < (char)0) ? -CHAR_MAX-1 : 0), "");
_Static_assert(CHAR_MAX == (((char)-1 < (char)0) ? -(CHAR_MIN+1) : (char)~0ULL), "");

#if __STDC_VERSION__ >= 199901 || __cplusplus >= 201103L
_Static_assert(LLONG_MAX == -(LLONG_MIN+1LL), "");
_Static_assert(LLONG_MIN == -LLONG_MAX-1LL, "");
_Static_assert(ULLONG_MAX == (unsigned long long)~0ULL, "");
#else
int LLONG_MIN, LLONG_MAX, ULLONG_MAX; // Not defined.
#endif

/* FIXME: This is using the placeholder dates Clang produces for these macros
   in C2x mode; switch to the correct values once they've been published. */
#if __STDC_VERSION__ >= 202000L
/* Validate the standard requirements. */
_Static_assert(BOOL_WIDTH >= 1);

_Static_assert(CHAR_WIDTH == CHAR_BIT);
_Static_assert(CHAR_WIDTH / CHAR_BIT == sizeof(char));
_Static_assert(SCHAR_WIDTH == CHAR_BIT);
_Static_assert(SCHAR_WIDTH / CHAR_BIT == sizeof(signed char));
_Static_assert(UCHAR_WIDTH == CHAR_BIT);
_Static_assert(UCHAR_WIDTH / CHAR_BIT == sizeof(unsigned char));

_Static_assert(USHRT_WIDTH >= 16);
_Static_assert(USHRT_WIDTH / CHAR_BIT == sizeof(unsigned short));
_Static_assert(SHRT_WIDTH == USHRT_WIDTH);
_Static_assert(SHRT_WIDTH / CHAR_BIT == sizeof(signed short));

_Static_assert(UINT_WIDTH >= 16);
_Static_assert(UINT_WIDTH / CHAR_BIT == sizeof(unsigned int));
_Static_assert(INT_WIDTH == UINT_WIDTH);
_Static_assert(INT_WIDTH / CHAR_BIT == sizeof(signed int));

_Static_assert(ULONG_WIDTH >= 32);
_Static_assert(ULONG_WIDTH / CHAR_BIT == sizeof(unsigned long));
_Static_assert(LONG_WIDTH == ULONG_WIDTH);
_Static_assert(LONG_WIDTH / CHAR_BIT == sizeof(signed long));

_Static_assert(ULLONG_WIDTH >= 64);
_Static_assert(ULLONG_WIDTH / CHAR_BIT == sizeof(unsigned long long));
_Static_assert(LLONG_WIDTH == ULLONG_WIDTH);
_Static_assert(LLONG_WIDTH / CHAR_BIT == sizeof(signed long long));

_Static_assert(BITINT_MAXWIDTH >= ULLONG_WIDTH);
#else
/* None of these are defined. */
int BOOL_WIDTH, CHAR_WIDTH, SCHAR_WIDTH, UCHAR_WIDTH, USHRT_WIDTH, SHRT_WIDTH,
    UINT_WIDTH, INT_WIDTH, ULONG_WIDTH, LONG_WIDTH, ULLONG_WIDTH, LLONG_WIDTH,
    BITINT_MAXWIDTH;
#endif
