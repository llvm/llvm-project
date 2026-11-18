// Test that LSan's atexit leak check does not scan stale pointers left in a
// dead user stack frame after exit() has been called.
//
// This is intentionally x86_64/glibc-specific. The inline assembly writes the
// leaked pointer into a known stack slot in a noinline helper frame. That frame
// is dead before main calls exit(0). Without the exit-entry stack boundary, the
// later DoLeakCheck stack pointer can make LSan scan the stale helper frame and
// incorrectly mark the allocation reachable.
//
// RUN: %clangxx_lsan -O0 -fno-omit-frame-pointer -DEXPLICIT_EXIT %s -o %t
// RUN: %env_lsan_opts="use_registers=0:use_stacks=1" not %run %t 2>&1 | FileCheck %s
//
// REQUIRES: x86_64-target-arch, glibc
// UNSUPPORTED: hwasan

#include <stdlib.h>

// This test detects a false negative caused by stale stack roots after exit().
// OFF must point to a stack-frame hole that survives until LSan's atexit leak
// check. The exact holes depend on the libc version and generated exit-handler
// stack layout. On glibc 2.39 (Ubuntu 24.04), known working values include 64,
// 80, 88, 160, 384, 400, 408, and 416.
#ifndef OFF
#  define OFF 64
#endif

#ifdef EXPLICIT_EXIT
__attribute__((noinline)) void leak_to_dead_frame(void) {
  __asm__ __volatile__("movl $100, %%edi\n\t"
                       "callq malloc@PLT\n\t"
                       "movq %%rax, -%c0(%%rbp)\n\t"
                       "xorq %%rax, %%rax\n\t"
                       :
                       : "i"(OFF)
                       : "rax", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10",
                         "r11", "memory");
}

int main(void) {
  leak_to_dead_frame();
  // Explicit exit called by user code.
  exit(0);
}

#else

int main(void) {
  __asm__ __volatile__("movl $100, %%edi\n\t"
                       "callq malloc@PLT\n\t"
                       "movq %%rax, -%c0(%%rbp)\n\t"
                       "xorq %%rax, %%rax\n\t"
                       :
                       : "i"(OFF)
                       : "rax", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10",
                         "r11", "memory");
  // Implicit exit called by libc after main returns.
  return 0;
}
#endif

// CHECK: LeakSanitizer: detected memory leaks
// CHECK: Direct leak of 100 byte(s) in 1 object(s)
// CHECK: SUMMARY: {{.*}}Sanitizer: 100 byte(s) leaked in 1 allocation(s)
