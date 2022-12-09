// RUN: %clang_cc1 -fsyntax-only -verify %s -include %S/Inputs/include.h

#include "Inputs/include.h"
#include "Inputs/include.h"

#define FOO(x) x + x
int k = FOO(FOO(123));
bool b = EQUALS(k, k);

#pragma clang __debug sloc_usage // expected-remark {{address space usage}}
// expected-note@* {{(0% of available space)}}
// (this file)     expected-note-re@1 {{file entered 1 time using {{.*}}B of space plus 51B for macro expansions}}
// (included file) expected-note-re@Inputs/include.h:1 {{file entered 3 times using {{.*}}B of space{{$}}}}
// (builtins file) expected-note@* {{file entered}}
