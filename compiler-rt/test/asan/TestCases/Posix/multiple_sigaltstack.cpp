// RUN: %clangxx_asan %s -o %t && %env_asan_opts=use_sigaltstack=1 %run %t

#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>

#if (__APPLE__)
char global_alt_stack[MINSIGSTKSZ];
#else
char global_alt_stack[4096 * 4];
#endif

int main() {
  stack_t altstack;
  altstack.ss_sp = global_alt_stack;
  altstack.ss_size = sizeof(global_alt_stack);
  altstack.ss_flags = 0;
  if (sigaltstack(&altstack, nullptr) != 0) {
    perror("sigaltstack");
    exit(1);
  }

  // UnsetAlternateSignalStack will get called when the thread exists. If we
  // don't *only* unmap a signal stack the runtime owns, we'll get a fault on
  // the munmap operation, since that memory isn't mmaped.
  return 0;
}
