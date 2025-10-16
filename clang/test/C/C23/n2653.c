// RUN: %clang_cc1 -ffreestanding -verify=c23 -std=c23 %s
// RUN: %clang_cc1 -ffreestanding -verify=c17 -std=c17 %s

// c23-no-diagnostics

#include <stdatomic.h>

#define __enable_constant_folding(x) (__builtin_constant_p(x) ? (x) : (x))
#define __is_same(a, b) (__extension__ _Generic(a, b: 1, default: 0) && __extension__ _Generic(b, a: 1, default: 0))

#ifndef ATOMIC_CHAR8_T_LOCK_FREE
#error missing
#endif
// c17-error@-2 {{missing}}

_Static_assert(__is_same(atomic_char8_t, unsigned char _Atomic), "");
// c17-error@-1 {{use of undeclared identifier 'atomic_char8_t'}}
// c17-error@-2 {{unknown type name 'atomic_char8_t'}}

_Static_assert(_Generic(u8"", unsigned char*: 1, char*: 0), "");
// c17-error@-1 {{static assertion failed}}

// -fsigned-char is the default
#define M(X) __enable_constant_folding((X) >= 0x80)

_Static_assert(M(u8"\U000000E9"[0]), "");
// c17-error@-1 {{static assertion failed}}
#if __STDC_VERSION__ >= 202311L
_Static_assert(M(u8'\xC3'), "");
#endif

const          char cu8[]  = u8"text";
const signed   char scu8[] = u8"text";
const unsigned char ucu8[] = u8"text";
