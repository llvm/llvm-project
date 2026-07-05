// RUN: %clang_cc1 -verify -Wall -Wextra -Wunused-macros -E -frewrite-includes %s
// expected-no-diagnostics

#pragma GCC visibility push (default)

#define USED_MACRO 1
int test() { return USED_MACRO; }
