// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

// NULL is rdefined in stddef.h
#define NULL ((void*) 0)

// These are headers bundled with Clang.
#include <stdarg.h>
#include <stddef.h>

#ifndef _VA_LIST
typedef __builtin_va_list va_list;
#endif

#ifndef _SIZE_T
typedef __typeof__(sizeof(int)) size_t;
#endif

#ifndef _WCHAR_T
typedef __typeof__(*L"") wchar_t;
#endif

extern void foo(wchar_t x);
extern void bar(size_t x);
void *baz() { return NULL; }
void quz() {
  va_list y;
}

