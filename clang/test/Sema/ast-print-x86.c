// RUN: %clang_cc1 -triple i686-elf %s -ast-print | FileCheck %s

// REQUIRES: x86-registered-target

void assembly(void) {
  int added;
  // CHECK: asm volatile ("addl %%ebx,%%eax" : "=a" (added) : "a" (1), "b" (2));
  __asm__ __volatile__("addl %%ebx,%%eax" : "=a" (added) : "a" (1), "b" (2) );
}
