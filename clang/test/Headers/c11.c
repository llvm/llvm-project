// RUN: rm -rf %t
// RUN: %clang_cc1 -fsyntax-only -verify -std=c11 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c11 -fmodules -fmodules-cache-path=%t %s -D__STDC_WANT_LIB_EXT1__=1
// RUN: %clang_cc1 -fsyntax-only -verify -std=c11 -ffreestanding %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c11 -triple i686-pc-win32 -fms-compatibility-version=17.00 %s

noreturn int f(); // expected-error 1+{{}}

#include <stdnoreturn.h>
#include <stdnoreturn.h>
#include <stdnoreturn.h>

int g();
noreturn int g();
int noreturn g();
int g();

#include <stdalign.h>
_Static_assert(__alignas_is_defined, "");
_Static_assert(__alignof_is_defined, "");
alignas(alignof(int)) char c[4];
_Static_assert(__alignof(c) == 4, "");

#define __STDC_WANT_LIB_EXT1__ 1
#include <stddef.h>
rsize_t x = 0;
_Static_assert(sizeof(max_align_t) >= sizeof(long long), "");
_Static_assert(alignof(max_align_t) >= alignof(long long), "");
_Static_assert(sizeof(max_align_t) >= sizeof(long double), "");
_Static_assert(alignof(max_align_t) >= alignof(long double), "");

#ifdef _MSC_VER
_Static_assert(sizeof(max_align_t) == sizeof(double), "");
#endif

// If we are freestanding, then also check RSIZE_MAX (in a hosted implementation
// we will use the host stdint.h, which may not yet have C11 support).
#ifndef __STDC_HOSTED__
#include <stdint.h>
rsize_t x2 = RSIZE_MAX;
#endif

