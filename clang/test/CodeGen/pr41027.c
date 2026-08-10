// REQUIRES: x86-registered-target
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -S -O2 -o - %s | FileCheck %s

// CHECK-LABEL: f:
// CHECK:         movl $1, %eax
// CHECK-NEXT:    #APP
// CHECK-NEXT:    outl %eax, $1
// CHECK-NEXT:    #NO_APP

static inline void pr41027(unsigned a, unsigned b) {
  if (__builtin_constant_p(a)) {
    __asm__ volatile("outl %0,%w1" : : "a"(b), "n"(a));
  } else {
    __asm__ volatile("outl %0,%w1" : : "a"(b), "d"(a));
  }
}

void f(unsigned port) {
  pr41027(1, 1);
}
