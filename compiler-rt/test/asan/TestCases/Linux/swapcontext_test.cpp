// Check that ASan plays well with easy cases of makecontext/swapcontext.

// RUN: %clangxx_asan -O0 %s -o %t && %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -O2 %s -o %t && %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -O3 %s -o %t && %run %t 2>&1 | FileCheck %s
//
// This test is too sublte to try on non-x86 arch for now.
// Android and musl do not support swapcontext.
// REQUIRES: x86-target-arch && glibc-2.27

#include <assert.h>
#include <memory.h>
#include <stdio.h>
#include <ucontext.h>
#include <unistd.h>

ucontext_t orig_context;
ucontext_t child_context;

const int kStackSize = 1 << 20;

__attribute__((noinline))
void Throw() {
  throw 1;
}

__attribute__((noinline))
void ThrowAndCatch() {
  try {
    Throw();
  } catch(int a) {
    printf("ThrowAndCatch: %d\n", a);
  }
}

void Child(int mode) {
  assert(orig_context.uc_stack.ss_size == 0);
  char x[32] = {0};  // Stack gets poisoned.
  printf("Child: %p\n", x);
  ThrowAndCatch();  // Simulate __asan_handle_no_return().
  // (a) Do nothing, just return to parent function.
  // (b) Jump into the original function. Stack remains poisoned unless we do
  //     something.
  if (mode == 1) {
    if (swapcontext(&child_context, &orig_context) < 0) {
      perror("swapcontext");
      _exit(0);
    }
  }
}

int Run(int arg, int mode, char *child_stack) {
  printf("Child stack: %p\n", child_stack);
  // Setup child context.
  memset(&child_context, 0xff, sizeof(child_context));
  getcontext(&child_context);
  assert(child_context.uc_stack.ss_size == 0);
  child_context.uc_stack.ss_sp = child_stack;
  child_context.uc_stack.ss_size = kStackSize / 2;
  if (mode == 0) {
    child_context.uc_link = &orig_context;
  }
  makecontext(&child_context, (void (*)())Child, 1, mode);
  memset(&orig_context, 0xff, sizeof(orig_context));
  if (swapcontext(&orig_context, &child_context) < 0) {
    perror("swapcontext");
    return 0;
  }
  // Touch childs's stack to make sure it's unpoisoned.
  for (int i = 0; i < kStackSize; i++) {
    child_stack[i] = i;
  }
  return child_stack[arg];
}

int main(int argc, char **argv) {
  char stack[kStackSize + 1];
  // CHECK: WARNING: ASan doesn't fully support makecontext/swapcontext
  int ret = 0;
  ret += Run(argc - 1, 0, stack);
  printf("Test1 passed\n");
  // CHECK: Test1 passed
  ret += Run(argc - 1, 1, stack);
  printf("Test2 passed\n");
  // CHECK: Test2 passed
  char *heap = new char[kStackSize + 1];
  ret += Run(argc - 1, 0, heap);
  printf("Test3 passed\n");
  // CHECK: Test3 passed
  ret += Run(argc - 1, 1, heap);
  printf("Test4 passed\n");
  // CHECK: Test4 passed

  delete [] heap;
  return ret;
}
