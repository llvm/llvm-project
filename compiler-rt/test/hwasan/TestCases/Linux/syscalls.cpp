// RUN: %clangxx_hwasan -O0 %s -o %t && not %env_hwasan_opts=symbolize=0 %run %t 2>&1 | FileCheck %s --implicit-check-not=RETURN_FROM_TEST
// RUN: %clangxx_hwasan -O3 %s -o %t && not %env_hwasan_opts=symbolize=0 %run %t 2>&1 | FileCheck %s --implicit-check-not=RETURN_FROM_TEST
// RUN: %clangxx_hwasan -O0 %s -o %t && not %env_hwasan_opts=halt_on_error=0:symbolize=0 %run %t 2>&1 | FileCheck %s --implicit-check-not=RETURN_FROM_TEST --check-prefixes=CHECK,RECOVER

// UNSUPPORTED: android

#include <assert.h>
#include <errno.h>
#include <glob.h>
#include <malloc.h>
#include <stdio.h>
#include <string.h>

#include <sanitizer/hwasan_interface.h>
#include <sanitizer/linux_syscall_hooks.h>

/* Test the presence of __sanitizer_syscall_ in the tool runtime, and general
   sanity of their behaviour. */

int main(int argc, char *argv[]) {
  // lit.cfg.py currently sets 'disable_allocator_tagging=1'
  __hwasan_enable_allocator_tagging();

  char *buf = (char *)malloc(1000);
  assert(buf != NULL);

  __sanitizer_syscall_pre_recvmsg(0, buf - 1, 0);
  // CHECK: HWAddressSanitizer: tag-mismatch on address [[PTR:0x[a-f0-9]+]]
  // CHECK: Cause: heap-buffer-overflow
  // CHECK: [[PTR]] is located 1 bytes before a 1000-byte region

  free(buf);
  fprintf(stderr, "RETURN_FROM_TEST\n");
  // RECOVER: RETURN_FROM_TEST
  return 0;
}
