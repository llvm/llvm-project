// RUN: %clang_cc1 -triple x86_64 -fsyntax-only %s
// XFAIL: *

inline void pr41027(unsigned a, unsigned b) {
  if (__builtin_constant_p(a)) {
    __asm__ volatile("outl %0,%w1" : : "a"(b), "n"(a));
  } else {
    __asm__ volatile("outl %0,%w1" : : "a"(b), "d"(a));
  }
}
