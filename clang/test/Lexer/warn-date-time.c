// RUN: %clang_cc1 -Wdate-time -Wno-builtin-macro-redefined %s -verify -E
// RUN: %clang_cc1 -Wdate-time -Wno-builtin-macro-redefined %s -DIS_SYSHEADER -verify -E
// RUN: not %clang_cc1 -Werror=date-time -Wno-builtin-macro-redefined %s -DIS_SYSHEADER -E 2>&1 | grep 'error: expansion' | count 3


#ifdef IS_HEADER

#ifdef IS_SYSHEADER
#pragma clang system_header
#endif

__TIME__ // expected-warning {{expansion of date or time macro is not reproducible}}
__DATE__  // expected-warning {{expansion of date or time macro is not reproducible}}
__TIMESTAMP__ // expected-warning {{expansion of date or time macro is not reproducible}}

#define __TIME__
__TIME__

#else

#define IS_HEADER
#include __FILE__
#endif
