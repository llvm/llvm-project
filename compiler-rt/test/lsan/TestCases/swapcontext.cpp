// We can't unwind stack if we're running coroutines on heap-allocated
// memory. Make sure we don't report these leaks.

// RUN: %clangxx_lsan %s -o %t
// RUN: %env_lsan_opts= %run %t 2>&1
// RUN: %env_lsan_opts= not %run %t foo 2>&1 | FileCheck %s
// Missing 'getcontext' and 'makecontext' on Android.
// UNSUPPORTED: arm,aarch64,powerpc64,android

#include "sanitizer_common/sanitizer_ucontext.h"
#include <stdio.h>
#include <unistd.h>

const int kStackSize = 1 << 20;

void Child() {
  int child_stack;
  printf("Child: %p\n", &child_stack);
  int *leaked = new int[666];
}

int main(int argc, char *argv[]) {
  char stack_memory[kStackSize + 1] __attribute__((aligned(16)));
  char *heap_memory = new char[kStackSize + 1];
  char *child_stack = (argc > 1) ? stack_memory : heap_memory;

  printf("Child stack: %p\n", child_stack);
  ucontext_t orig_context;
  ucontext_t child_context;
  getcontext(&child_context);
  child_context.uc_stack.ss_sp = child_stack;
  child_context.uc_stack.ss_size = kStackSize / 2;
  child_context.uc_link = &orig_context;
  makecontext(&child_context, Child, 0);
  if (swapcontext(&orig_context, &child_context) < 0) {
    perror("swapcontext");
    return 1;
  }

  delete[] heap_memory;
  return 0;
}

// CHECK: SUMMARY: {{(Leak|Address)}}Sanitizer: 2664 byte(s) leaked in 1 allocation(s)
