// RUN: %clang_cc1 -fsyntax-only -ffreestanding %s -verify
// RUN: %clang_cc1 -fsyntax-only -ffreestanding -x c++ %s -verify
// expected-no-diagnostics

#if defined(i386) || defined(__x86_64__)
#include <mm3dnow.h>

int foo(void *x) {
  _m_prefetch(x);
  _m_prefetchw(x);
  return 4;
}
#endif
