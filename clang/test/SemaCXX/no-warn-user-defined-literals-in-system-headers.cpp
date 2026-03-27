// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 -isystem %S %s

#include <no-warn-user-defined-literals-in-system-headers.h>

void operator "" bar(long double); // expected-warning{{user-defined literal suffixes not starting with '_' are reserved}}
