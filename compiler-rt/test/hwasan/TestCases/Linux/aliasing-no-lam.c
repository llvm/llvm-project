// Test that aliasing mode starts up with fail_without_syscall_abi=1 and does
// not enable x86 LAM for the process.
//
// RUN: %clang_hwasan %s -o %t
// RUN: %env_hwasan_opts=fail_without_syscall_abi=1 %run %t 2>&1 | FileCheck %s
//
// REQUIRES: x86_64-target-arch
// UNSUPPORTED: pointer-tagging

#include <stdio.h>
#include <sys/syscall.h>
#include <unistd.h>

#define ARCH_GET_UNTAG_MASK 0x4001

int main() {
  unsigned long mask = 0;
  // Kernels without LAM support reject ARCH_GET_UNTAG_MASK; nothing to check.
  if (syscall(SYS_arch_prctl, ARCH_GET_UNTAG_MASK, &mask) == 0 &&
      mask != ~0UL) {
    fprintf(stderr, "LAM enabled, untag mask: %#lx\n", mask);
    return 1;
  }
  fprintf(stderr, "DONE\n");
  return 0;
}

// CHECK-NOT: FATAL
// CHECK: DONE
