// RUN: %clang_cc1 %s -fsyntax-only -verify

#define FOO foo>
#include <:FOO
// expected-error@-1 {{expected "FILENAME" or <FILENAME>}}
