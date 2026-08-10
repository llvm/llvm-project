// RUN: %clang_cc1 -ffreestanding -fsyntax-only -verify -std=c17 %s
// RUN: %clang_cc1 -ffreestanding -fsyntax-only -verify -std=c2x %s
// RUN: %clang_cc1 %s -ffreestanding -std=c2x -fsyntax-only -triple=aarch64-none-none
// RUN: %clang_cc1 %s -ffreestanding -std=c2x -fsyntax-only -triple=arm-none-none
// RUN: %clang_cc1 %s -ffreestanding -std=c2x -fsyntax-only -triple=i386-none-none
// RUN: %clang_cc1 %s -ffreestanding -std=c2x -fsyntax-only -triple=mips-none-none
// RUN: %clang_cc1 %s -ffreestanding -std=c2x -fsyntax-only -triple=mips64-none-none
// RUN: %clang_cc1 %s -ffreestanding -std=c2x -fsyntax-only -triple=msp430-none-none
// RUN: %clang_cc1 %s -ffreestanding -std=c2x -fsyntax-only -triple=powerpc64-none-none
// RUN: %clang_cc1 %s -ffreestanding -std=c2x -fsyntax-only -triple=powerpc64-none-netbsd
// RUN: %clang_cc1 %s -ffreestanding -std=c2x -fsyntax-only -triple=powerpc-none-none
// RUN: %clang_cc1 %s -ffreestanding -std=c2x -fsyntax-only -triple=s390x-none-none
// RUN: %clang_cc1 %s -ffreestanding -std=c2x -fsyntax-only -triple=sparc-none-none
// RUN: %clang_cc1 %s -ffreestanding -std=c2x -fsyntax-only -triple=tce-none-none
// RUN: %clang_cc1 %s -ffreestanding -std=c2x -fsyntax-only -triple=x86_64-none-none
// RUN: %clang_cc1 %s -ffreestanding -std=c2x -fsyntax-only -triple=x86_64-pc-linux-gnu
// RUN: %clang_cc1 %s -ffreestanding -std=c2x -fsyntax-only -triple=i386-mingw32
// RUN: %clang_cc1 %s -ffreestanding -std=c2x -fsyntax-only -triple=xcore-none-none
// expected-no-diagnostics

#include <stdint.h>

#if __STDC_VERSION__ >= 202311L
/* Validate the standard requirements. */
_Static_assert(SIG_ATOMIC_WIDTH >= 8);
_Static_assert(SIZE_WIDTH >= 16);
_Static_assert(SIZE_WIDTH / __CHAR_BIT__ == sizeof(sizeof(0)));
_Static_assert(WCHAR_WIDTH >= 8);
_Static_assert(WCHAR_WIDTH / __CHAR_BIT__ == sizeof(L't'));
_Static_assert(WINT_WIDTH >= 16);
_Static_assert(UINTPTR_WIDTH >= 16);
_Static_assert(UINTPTR_WIDTH / __CHAR_BIT__ == sizeof(uintptr_t));
_Static_assert(INTPTR_WIDTH == UINTPTR_WIDTH);
_Static_assert(INTPTR_WIDTH / __CHAR_BIT__ == sizeof(intptr_t));

/* FIXME: the TCE target is not a conforming C target because it defines these
   values to be less than 64. */
#if !defined(__TCE__)
_Static_assert(UINTMAX_WIDTH >= 64);
_Static_assert(UINTMAX_WIDTH / __CHAR_BIT__ == sizeof(uintmax_t));
_Static_assert(INTMAX_WIDTH == UINTMAX_WIDTH);
_Static_assert(INTMAX_WIDTH / __CHAR_BIT__ == sizeof(intmax_t));
#endif

/* NB: WG14 N2412 set this to 17, but WG14 N2808 set it back to 16. */
_Static_assert(PTRDIFF_WIDTH >= 16);
#else
/* None of these are defined. */
int PTRDIFF_WIDTH, SIG_ATOMIC_WIDTH, SIZE_WIDTH, WCHAR_WIDTH, WINT_WIDTH,
    INTPTR_WIDTH, UINTPTR_WIDTH, INTMAX_WIDTH, UINTMAX_WIDTH;
#endif

#if defined(INT8_MAX) && __STDC_VERSION__ >= 202311L
_Static_assert(INT8_WIDTH == 8, "");
_Static_assert(UINT8_WIDTH == INT8_WIDTH, "");
_Static_assert(INT8_WIDTH / __CHAR_BIT__ == sizeof(int8_t), "");
_Static_assert(UINT8_WIDTH / __CHAR_BIT__ == sizeof(uint8_t), "");
#else
int INT8_WIDTH, UINT8_WIDTH; /* None of these are defined. */
#endif
#if defined(INT_LEAST8_MAX) && __STDC_VERSION__ >= 202311L
_Static_assert(INT_LEAST8_WIDTH >= 8, "");
_Static_assert(INT_LEAST8_WIDTH / __CHAR_BIT__ == sizeof(int_least8_t), "");
_Static_assert(UINT_LEAST8_WIDTH == INT_LEAST8_WIDTH, "");
_Static_assert(UINT_LEAST8_WIDTH / __CHAR_BIT__ == sizeof(uint_least8_t), "");
#else
int INT_LEAST8_WIDTH, UINT_LEAST8_WIDTH; /* None of these are defined. */
#endif
#if defined(INT_FAST8_MAX) && __STDC_VERSION__ >= 202311L
_Static_assert(INT_FAST8_WIDTH >= 8, "");
_Static_assert(INT_FAST8_WIDTH / __CHAR_BIT__ == sizeof(int_fast8_t), "");
_Static_assert(UINT_FAST8_WIDTH == INT_FAST8_WIDTH, "");
_Static_assert(UINT_FAST8_WIDTH / __CHAR_BIT__ == sizeof(uint_fast8_t), "");
#else
int INT_FAST8_WIDTH, UINT_FAST8_WIDTH; /* None of these are defined. */
#endif

#if defined(INT16_MAX) && __STDC_VERSION__ >= 202311L
_Static_assert(INT16_WIDTH == 16, "");
_Static_assert(UINT16_WIDTH == INT16_WIDTH, "");
_Static_assert(INT16_WIDTH / __CHAR_BIT__ == sizeof(int16_t), "");
_Static_assert(UINT16_WIDTH / __CHAR_BIT__ == sizeof(uint16_t), "");
#else
int INT16_WIDTH, UINT16_WIDTH; /* None of these are defined. */
#endif
#if defined(INT_LEAST16_MAX) && __STDC_VERSION__ >= 202311L
_Static_assert(INT_LEAST16_WIDTH >= 16, "");
_Static_assert(INT_LEAST16_WIDTH / __CHAR_BIT__ == sizeof(int_least16_t), "");
_Static_assert(UINT_LEAST16_WIDTH == INT_LEAST16_WIDTH, "");
_Static_assert(UINT_LEAST16_WIDTH / __CHAR_BIT__ == sizeof(uint_least16_t), "");
#else
int INT_LEAST16_WIDTH, UINT_LEAST16_WIDTH; /* None of these are defined. */
#endif
#if defined(INT_FAST16_MAX) && __STDC_VERSION__ >= 202311L
_Static_assert(INT_FAST16_WIDTH >= 16, "");
_Static_assert(INT_FAST16_WIDTH / __CHAR_BIT__ == sizeof(int_fast16_t), "");
_Static_assert(UINT_FAST16_WIDTH == INT_FAST16_WIDTH, "");
_Static_assert(UINT_FAST16_WIDTH / __CHAR_BIT__ == sizeof(int_fast16_t), "");
#else
int INT_FAST16_WIDTH, UINT_FAST16_WIDTH; /* None of these are defined. */
#endif

#if defined(INT24_MAX) && __STDC_VERSION__ >= 202311L
_Static_assert(INT24_WIDTH == 24, "");
_Static_assert(UINT24_WIDTH == INT24_WIDTH, "");
_Static_assert(INT24_WIDTH / __CHAR_BIT__ == sizeof(int24_t), "");
_Static_assert(UINT24_WIDTH / __CHAR_BIT__ == sizeof(uint24_t), "");
#else
int INT24_WIDTH, UINT24_WIDTH; /* None of these are defined. */
#endif
#if defined(INT_LEAST24_MAX) && __STDC_VERSION__ >= 202311L
_Static_assert(INT_LEAST24_WIDTH >= 24, "");
_Static_assert(INT_LEAST24_WIDTH / __CHAR_BIT__ == sizeof(int_least24_t), "");
_Static_assert(UINT_LEAST24_WIDTH == INT_LEAST24_WIDTH, "");
_Static_assert(UINT_LEAST24_WIDTH / __CHAR_BIT__ == sizeof(uint_least24_t), "");
#else
int INT_LEAST24_WIDTH, UINT_LEAST24_WIDTH; /* None of these are defined. */
#endif
#if defined(INT_FAST24_MAX) && __STDC_VERSION__ >= 202311L
_Static_assert(INT_FAST24_WIDTH >= 24, "");
_Static_assert(INT_FAST24_WIDTH / __CHAR_BIT__ == sizeof(int_fast24_t), "");
_Static_assert(UINT_FAST24_WIDTH == INT_FAST24_WIDTH, "");
_Static_assert(UINT_FAST24_WIDTH / __CHAR_BIT__ == sizeof(uint_fast24_t), "");
#else
int INT_FAST24_WIDTH, UINT_FAST24_WIDTH; /* None of these are defined. */
#endif

#if defined(INT32_MAX) && __STDC_VERSION__ >= 202311L
_Static_assert(INT32_WIDTH == 32, "");
_Static_assert(UINT32_WIDTH == INT32_WIDTH, "");
_Static_assert(INT32_WIDTH / __CHAR_BIT__ == sizeof(int32_t), "");
_Static_assert(UINT32_WIDTH / __CHAR_BIT__ == sizeof(uint32_t), "");
#else
int INT32_WIDTH, UINT32_WIDTH; /* None of these are defined. */
#endif
#if defined(INT_LEAST32_MAX) && __STDC_VERSION__ >= 202311L
_Static_assert(INT_LEAST32_WIDTH >= 32, "");
_Static_assert(INT_LEAST32_WIDTH / __CHAR_BIT__ == sizeof(int_least32_t), "");
_Static_assert(UINT_LEAST32_WIDTH == INT_LEAST32_WIDTH, "");
_Static_assert(UINT_LEAST32_WIDTH / __CHAR_BIT__ == sizeof(uint_least32_t), "");
#else
int INT_LEAST32_WIDTH, UINT_LEAST32_WIDTH; /* None of these are defined. */
#endif
#if defined(INT_FAST32_MAX) && __STDC_VERSION__ >= 202311L
_Static_assert(INT_FAST32_WIDTH >= 32, "");
_Static_assert(INT_FAST32_WIDTH / __CHAR_BIT__ == sizeof(int_fast32_t), "");
_Static_assert(UINT_FAST32_WIDTH == INT_FAST32_WIDTH, "");
_Static_assert(UINT_FAST32_WIDTH / __CHAR_BIT__ == sizeof(uint_fast32_t), "");
#else
int INT_FAST32_WIDTH, UINT_FAST32_WIDTH; /* None of these are defined. */
#endif

#if defined(INT40_MAX) && __STDC_VERSION__ >= 202311L
_Static_assert(INT40_WIDTH == 40, "");
_Static_assert(UINT40_WIDTH == INT40_WIDTH, "");
_Static_assert(INT40_WIDTH / __CHAR_BIT__ == sizeof(int40_t), "");
_Static_assert(UINT40_WIDTH / __CHAR_BIT__ == sizeof(uint40_t), "");
#else
int INT40_WIDTH, UINT40_WIDTH; /* None of these are defined. */
#endif
#if defined(INT_LEAST40_MAX) && __STDC_VERSION__ >= 202311L
_Static_assert(INT_LEAST40_WIDTH >= 40, "");
_Static_assert(INT_LEAST40_WIDTH / __CHAR_BIT__ == sizeof(int_least40_t), "");
_Static_assert(UINT_LEAST40_WIDTH == INT_LEAST40_WIDTH, "");
_Static_assert(UINT_LEAST40_WIDTH / __CHAR_BIT__ == sizeof(int_least40_t), "");
#else
int INT_LEAST40_WIDTH, UINT_LEAST40_WIDTH; /* None of these are defined. */
#endif
#if defined(INT_FAST40_MAX) && __STDC_VERSION__ >= 202311L
_Static_assert(INT_FAST40_WIDTH >= 40, "");
_Static_assert(INT_FAST40_WIDTH / __CHAR_BIT__ == sizeof(int_fast40_t), "");
_Static_assert(UINT_FAST40_WIDTH == INT_FAST40_WIDTH, "");
_Static_assert(UINT_FAST40_WIDTH / __CHAR_BIT__ == sizeof(uint_fast40_t), "");
#else
int INT_FAST40_WIDTH, UINT_FAST40_WIDTH; /* None of these are defined. */
#endif

#if defined(INT48_MAX) && __STDC_VERSION__ >= 202311L
_Static_assert(INT48_WIDTH == 48, "");
_Static_assert(UINT48_WIDTH == INT48_WIDTH, "");
_Static_assert(INT48_WIDTH / __CHAR_BIT__ == sizeof(int48_t), "");
_Static_assert(UINT48_WIDTH / __CHAR_BIT__ == sizeof(uint48_t), "");
#else
int INT48_WIDTH, UINT48_WIDTH; /* None of these are defined. */
#endif
#if defined(INT_LEAST48_MAX) && __STDC_VERSION__ >= 202311L
_Static_assert(INT_LEAST48_WIDTH >= 48, "");
_Static_assert(INT_LEAST48_WIDTH / __CHAR_BIT__ == sizeof(int_least48_t), "");
_Static_assert(UINT_LEAST48_WIDTH == INT_LEAST48_WIDTH, "");
_Static_assert(UINT_LEAST48_WIDTH / __CHAR_BIT__ == sizeof(int_least48_t), "");
#else
int INT_LEAST48_WIDTH, UINT_LEAST48_WIDTH; /* None of these are defined. */
#endif
#if defined(INT_FAST48_MAX) && __STDC_VERSION__ >= 202311L
_Static_assert(INT_FAST48_WIDTH >= 48, "");
_Static_assert(INT_FAST48_WIDTH / __CHAR_BIT__ == sizeof(int_fast48_t), "");
_Static_assert(UINT_FAST48_WIDTH == INT_FAST48_WIDTH, "");
_Static_assert(UINT_FAST48_WIDTH / __CHAR_BIT__ == sizeof(int_fast48_t), "");
#else
int INT_FAST48_WIDTH, UINT_FAST48_WIDTH; /* None of these are defined. */
#endif

#if defined(INT56_MAX) && __STDC_VERSION__ >= 202311L
_Static_assert(INT56_WIDTH == 56, "");
_Static_assert(UINT56_WIDTH == INT56_WIDTH, "");
_Static_assert(INT56_WIDTH / __CHAR_BIT__ == sizeof(int56_t), "");
_Static_assert(UINT56_WIDTH / __CHAR_BIT__ == sizeof(uint56_t), "");
#else
int INT56_WIDTH, UINT56_WIDTH; /* None of these are defined. */
#endif
#if defined(INT_LEAST56_MAX) && __STDC_VERSION__ >= 202311L
_Static_assert(INT_LEAST56_WIDTH >= 56, "");
_Static_assert(INT_LEAST56_WIDTH / __CHAR_BIT__ == sizeof(int_least56_t), "");
_Static_assert(UINT_LEAST56_WIDTH == INT_LEAST56_WIDTH, "");
_Static_assert(UINT_LEAST56_WIDTH / __CHAR_BIT__ == sizeof(int_least56_t), "");
#else
int INT_LEAST56_WIDTH, UINT_LEAST56_WIDTH; /* None of these are defined. */
#endif
#if defined(INT_FAST56_MAX) && __STDC_VERSION__ >= 202311L
_Static_assert(INT_FAST56_WIDTH >= 56, "");
_Static_assert(INT_FAST56_WIDTH / __CHAR_BIT__ == sizeof(int_fast56_t), "");
_Static_assert(UINT_FAST56_WIDTH == INT_FAST56_WIDTH, "");
_Static_assert(UINT_FAST56_WIDTH / __CHAR_BIT__ == sizeof(int_fast56_t), "");
#else
int INT_FAST56_WIDTH, UINT_FAST56_WIDTH; /* None of these are defined. */
#endif

#if defined(INT64_MAX) && __STDC_VERSION__ >= 202311L
_Static_assert(INT64_WIDTH == 64, "");
_Static_assert(UINT64_WIDTH == INT64_WIDTH, "");
_Static_assert(INT64_WIDTH / __CHAR_BIT__ == sizeof(int64_t), "");
_Static_assert(UINT64_WIDTH / __CHAR_BIT__ == sizeof(uint64_t), "");
#else
int INT64_WIDTH, UINT64_WIDTH; /* None of these are defined. */
#endif
#if defined(INT_LEAST64_MAX) && __STDC_VERSION__ >= 202311L
_Static_assert(INT_LEAST64_WIDTH >= 64, "");
_Static_assert(INT_LEAST64_WIDTH / __CHAR_BIT__ == sizeof(int_least64_t), "");
_Static_assert(UINT_LEAST64_WIDTH == INT_LEAST64_WIDTH, "");
_Static_assert(UINT_LEAST64_WIDTH / __CHAR_BIT__ == sizeof(int_least64_t), "");
#else
int INT_LEAST64_WIDTH, UINT_LEAST64_WIDTH; /* None of these are defined. */
#endif
#if defined(INT_FAST64_MAX) && __STDC_VERSION__ >= 202311L
_Static_assert(INT_FAST64_WIDTH >= 64, "");
_Static_assert(INT_FAST64_WIDTH / __CHAR_BIT__ == sizeof(int_fast64_t), "");
_Static_assert(UINT_FAST64_WIDTH == INT_FAST64_WIDTH, "");
_Static_assert(UINT_FAST64_WIDTH / __CHAR_BIT__ == sizeof(int_fast64_t), "");
#else
int INT_FAST64_WIDTH, UINT_FAST64_WIDTH; /* None of these are defined. */
#endif
