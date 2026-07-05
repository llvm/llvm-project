// Check that ASan plays well with easy cases of makecontext/swapcontext.

// RUN: %clangxx_asan -O0 %s -o %t && %run %t
// RUN: %clangxx_asan -O3 %s -o %t && %run %t
// RUN: %clangxx_asan -fsanitize-address-use-after-return=never -O0 %s -o %t && %run %t
// RUN: %clangxx_asan -fsanitize-address-use-after-return=never -O3 %s -o %t && %run %t
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

__attribute__((noinline)) void Throw() { throw 1; }

__attribute__((noinline)) void ThrowAndCatch() {
  try {
    Throw();
  } catch (int a) {
    printf("ThrowAndCatch: %d\n", a);
  }
}

void Child(int mode, int a, int b, int c) {
  char x[32] = {0}; // Stack gets poisoned.
  printf("Child: %d\n", x);
  assert(a == 'a');
  assert(b == 'b');
  assert(c == 'c');
  ThrowAndCatch(); // Simulate __asan_handle_no_return().
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
  getcontext(&child_context);
  child_context.uc_stack.ss_sp = child_stack;
  child_context.uc_stack.ss_size = kStackSize / 2;
  if (mode == 0) {
    child_context.uc_link = &orig_context;
  }
  makecontext(&child_context, (void (*)())Child, 4, mode, 'a', 'b', 'c');
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

ucontext_t poll_context;
ucontext_t poller_context;

void Poll() {
  swapcontext(&poll_context, &poller_context);

  {
    char x = 0;
    printf("POLL: %p\n", &x);
  }

  swapcontext(&poll_context, &poller_context);
}

void DoRunPoll(char *poll_stack) {
  getcontext(&poll_context);
  poll_context.uc_stack.ss_sp = poll_stack;
  poll_context.uc_stack.ss_size = kStackSize / 2;
  makecontext(&poll_context, Poll, 0);

  getcontext(&poller_context);

  swapcontext(&poller_context, &poll_context);
  swapcontext(&poller_context, &poll_context);

  // Touch poll's stack to make sure it's unpoisoned.
  for (int i = 0; i < kStackSize; i++) {
    poll_stack[i] = i;
  }
}

void RunPoll() {
  char *poll_stack = new char[kStackSize];

  for (size_t i = 0; i < 2; ++i) {
    DoRunPoll(poll_stack);
  }

  delete[] poll_stack;
}

int main(int argc, char **argv) {
  char stack[kStackSize + 1];
  int ret = 0;
  ret += Run(argc - 1, 0, stack);
  ret += Run(argc - 1, 1, stack);
  char *heap = new char[kStackSize + 1];
  ret += Run(argc - 1, 0, heap);
  ret += Run(argc - 1, 1, heap);

  RunPoll();
  delete[] heap;
  return ret;
}
