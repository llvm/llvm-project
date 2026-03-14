// RUN: %clang_cc1 -triple mips64-unknown-freebsd -std=c11 -verify %s
// RUN: %clang_cc1 -triple mips-unknown-freebsd -std=c11 -verify %s
// RUN: %clang_cc1 -triple mips-unknown-linux -std=c11 -verify %s
// RUN: %clang_cc1 -triple mips64-unknown-linux -std=c11 -verify %s
// expected-no-diagnostics

#ifdef _ABI64
#  ifdef __FreeBSD__
_Static_assert(sizeof(long double) == 8, "sizeof long double is wrong");
_Static_assert(_Alignof(long double) == 8, "alignof long double is wrong");
#  else
_Static_assert(sizeof(long double) == 16, "sizeof long double is wrong");
_Static_assert(_Alignof(long double) == 16, "alignof long double is wrong");
#  endif
#else
_Static_assert(sizeof(long double) == 8, "sizeof long double is wrong");
_Static_assert(_Alignof(long double) == 8, "alignof long double is wrong");
#endif

