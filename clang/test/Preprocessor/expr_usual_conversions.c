// RUN: %clang_cc1 %s -E -verify

#define INTMAX_MIN (-9223372036854775807LL -1)

#if (-42 + 0U) /* expected-warning {{left side of operator converted from negative value to unsigned: -42 to 18446744073709551574}} */  \
  / -2         /* expected-warning {{right side of operator converted from negative value to unsigned: -2 to 18446744073709551614}} */
foo
#endif

// Shifts don't want the usual conversions: PR2279
#if (2 << 1U) - 30 >= 0
#error
#endif

