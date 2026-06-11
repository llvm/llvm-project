// RUN: %clang_cl -fno-integrated-cc1 -E /experimental:deterministic /d1nodatetime -- %s
// RUN: %clang_cl -fno-integrated-cc1 -E /D IS_SYSHEADER=1 /experimental:deterministic /d1nodatetime -- %s

// RUN: %clang_cl -E /experimental:deterministic /d1nodatetime -- %s
// RUN: %clang_cl -E /D IS_SYSHEADER=1 /experimental:deterministic /d1nodatetime -- %s

// not %clang_cc1 -Werror=date-time -Wno-builtin-macro-redefined %s -DIS_SYSHEADER -E 2>&1 | grep 'error: expansion' | count 3

// RUN: %clang_cl -E -### /experimental:deterministic -- %s 2>&1 | FileCheck %s --check-prefix=WDATETIME
// WDATETIME: -Wdate-time
// WDATETIME: -mno-incremental-linker-compatible
// RUN: %clang_cl -E -### /d1nodatetime -- %s 2>&1 | FileCheck %s --check-prefix=MACROREDEF
// MACROREDEF: -init-datetime-macros=undefined

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
