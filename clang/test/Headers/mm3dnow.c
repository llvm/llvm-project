// RUN: %clang_cc1 -fsyntax-only -ffreestanding %s -verify
// RUN: %clang_cc1 -fsyntax-only -D_CLANG_DISABLE_CRT_DEPRECATION_WARNINGS -ffreestanding %s -verify
// RUN: %clang_cc1 -fsyntax-only -ffreestanding -x c++ %s -verify

// XFAIL: target=arm64ec-pc-windows-msvc
// These intrinsics are not yet implemented for Arm64EC.

#if defined(i386) || defined(__x86_64__)
#ifndef _CLANG_DISABLE_CRT_DEPRECATION_WARNINGS
// expected-warning@mm3dnow.h:*{{The <mm3dnow.h> header is deprecated}}
#else
// expected-no-diagnostics
#endif

#include <mm3dnow.h>

int foo(void *x) {
  _m_prefetch(x);
  _m_prefetchw(x);
  return 4;
}
#else
// expected-no-diagnostics
#endif
