// RUN: %clang_cc1 -fsyntax-only -ffreestanding %s -verify
// RUN: %clang_cc1 -fsyntax-only -ffreestanding -x c++ %s -verify
// expected-no-diagnostics

#if defined(i386) || defined(__x86_64__)
#include <mm3dnow.h>

int __attribute__((__target__(("3dnow")))) foo(int a) {
  _m_femms();
  return 4;
}

__m64 __attribute__((__target__(("3dnowa")))) bar(__m64 a) {
  return _m_pf2iw(a);
}
#endif
