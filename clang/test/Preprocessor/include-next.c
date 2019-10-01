// RUN: %clang_cc1 -verify %s -E -o /dev/null -I%S/Inputs/include-next-1 -I%S/Inputs/include-next-2 -DTEST=1
// RUN: %clang_cc1 -verify %s -E -o /dev/null -I%S/Inputs/include-next-1 -I%S/Inputs/include-next-2 -DTEST=2
// RUN: %clang_cc1 -verify %s -E -o /dev/null -I%S/Inputs/include-next-1 -I%S/Inputs/include-next-2 -DTEST=3

#if TEST == 1
// expected-warning@+1 {{#include_next in primary source file}}
#include_next "bar.h"
#if BAR != 1
#error wrong bar
#endif

#elif TEST == 2
// expected-no-diagnostics
#include "foo.h"
#if BAR != 2
#error wrong bar
#endif

#elif TEST == 3
// expected-warning@foo.h:1 {{#include_next in file found relative to primary source file or found by absolute path}}
#include "Inputs/include-next-1/foo.h"
#if BAR != 1
#error wrong bar
#endif
#undef BAR

#else
#error unknown test
#endif
