// Check that ASan plays well with annotated makecontext/swapcontext.

// RUN: %clangxx_asan -std=c++11 -lpthread -O0 %s -o %t && %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -std=c++11 -lpthread -O1 %s -o %t && %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -std=c++11 -lpthread -O2 %s -o %t && %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -std=c++11 -lpthread -O3 %s -o %t && %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -std=c++11 -lpthread -O0 %s -o %t && %run %t 2>&1 | FileCheck <( seq 60 | xargs -i -- grep LOOPCHECK %s ) --check-prefix LOOPCHECK
// RUN: %clangxx_asan -std=c++11 -lpthread -O1 %s -o %t && %run %t 2>&1 | FileCheck <( seq 60 | xargs -i -- grep LOOPCHECK %s ) --check-prefix LOOPCHECK
// RUN: %clangxx_asan -std=c++11 -lpthread -O2 %s -o %t && %run %t 2>&1 | FileCheck <( seq 60 | xargs -i -- grep LOOPCHECK %s ) --check-prefix LOOPCHECK
// RUN: %clangxx_asan -std=c++11 -lpthread -O3 %s -o %t && %run %t 2>&1 | FileCheck <( seq 60 | xargs -i -- grep LOOPCHECK %s ) --check-prefix LOOPCHECK

//
// This test is too subtle to try on non-x86 arch for now.
// REQUIRES: x86-target-arch

#include <pthread.h>
#include <setjmp.h>
#include <stdio.h>
#include <sys/time.h>
#include <ucontext.h>
#include <unistd.h>

#include <sanitizer/common_interface_defs.h>

ucontext_t orig_context;
ucontext_t child_context;
ucontext_t next_child_context;

char *next_child_stack;

const int kStackSize = 1 << 20;

const void *main_thread_stack;
size_t main_thread_stacksize;

const void *from_stack;
size_t from_stacksize;

__attribute__((noinline, noreturn)) void LongJump(jmp_buf env) {
  longjmp(env, 1);
  _exit(1);
}

// Simulate __asan_handle_no_return().
__attribute__((noinline)) void CallNoReturn() {
  jmp_buf env;
  if (setjmp(env) != 0) return;

  LongJump(env);
  _exit(1);
}

void NextChild() {
  CallNoReturn();
  __sanitizer_finish_switch_fiber(nullptr, &from_stack, &from_stacksize);

  printf("NextChild from: %p %zu\n", from_stack, from_stacksize);

  char x[32] = {0};  // Stack gets poisoned.
  printf("NextChild: %p\n", x);

  CallNoReturn();

  __sanitizer_start_switch_fiber(nullptr,
                                 main_thread_stack,
                                 main_thread_stacksize);
  CallNoReturn();
  if (swapcontext(&next_child_context, &orig_context) < 0) {
    perror("swapcontext");
    _exit(1);
  }
}

void Child(int mode) {
  CallNoReturn();
  __sanitizer_finish_switch_fiber(nullptr,
                                  &main_thread_stack,
                                  &main_thread_stacksize);
  char x[32] = {0};  // Stack gets poisoned.
  printf("Child: %p\n", x);
  CallNoReturn();
  // (a) Do nothing, just return to parent function.
  // (b) Jump into the original function. Stack remains poisoned unless we do
  //     something.
  // (c) Jump to another function which will then jump back to the main function
  if (mode == 0) {
    __sanitizer_start_switch_fiber(nullptr,
                                   main_thread_stack,
                                   main_thread_stacksize);
    CallNoReturn();
  } else if (mode == 1) {
    __sanitizer_start_switch_fiber(nullptr,
                                   main_thread_stack,
                                   main_thread_stacksize);
    CallNoReturn();
    if (swapcontext(&child_context, &orig_context) < 0) {
      perror("swapcontext");
      _exit(1);
    }
  } else if (mode == 2) {
    printf("NextChild stack: %p\n", next_child_stack);

    getcontext(&next_child_context);
    next_child_context.uc_stack.ss_sp = next_child_stack;
    next_child_context.uc_stack.ss_size = kStackSize / 2;
    makecontext(&next_child_context, (void (*)())NextChild, 0);
    __sanitizer_start_switch_fiber(nullptr,
                                   next_child_context.uc_stack.ss_sp,
                                   next_child_context.uc_stack.ss_size);
    CallNoReturn();
    if (swapcontext(&child_context, &next_child_context) < 0) {
      perror("swapcontext");
      _exit(1);
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
  makecontext(&child_context, (void (*)())Child, 1, mode);
  CallNoReturn();
  void* fake_stack_save;
  __sanitizer_start_switch_fiber(&fake_stack_save,
                                 child_context.uc_stack.ss_sp,
                                 child_context.uc_stack.ss_size);
  CallNoReturn();
  if (swapcontext(&orig_context, &child_context) < 0) {
    perror("swapcontext");
    _exit(1);
  }
  CallNoReturn();
  __sanitizer_finish_switch_fiber(fake_stack_save,
                                  &from_stack,
                                  &from_stacksize);
  CallNoReturn();
  printf("Main context from: %p %zu\n", from_stack, from_stacksize);

  // Touch childs's stack to make sure it's unpoisoned.
  for (int i = 0; i < kStackSize; i++) {
    child_stack[i] = i;
  }
  return child_stack[arg];
}

void handler(int sig) { CallNoReturn(); }

int main(int argc, char **argv) {
  // set up a signal that will spam and trigger __asan_handle_no_return at
  // tricky moments
  struct sigaction act = {};
  act.sa_handler = &handler;
  if (sigaction(SIGPROF, &act, 0)) {
    perror("sigaction");
    _exit(1);
  }

  itimerval t;
  t.it_interval.tv_sec = 0;
  t.it_interval.tv_usec = 10;
  t.it_value = t.it_interval;
  if (setitimer(ITIMER_PROF, &t, 0)) {
    perror("setitimer");
    _exit(1);
  }

  char *heap = new char[kStackSize + 1];
  next_child_stack = new char[kStackSize + 1];
  char stack[kStackSize + 1];
  // CHECK: WARNING: ASan doesn't fully support makecontext/swapcontext
  int ret = 0;
  // CHECK-NOT: ASan is ignoring requested __asan_handle_no_return
  for (unsigned int i = 0; i < 30; ++i) {
    ret += Run(argc - 1, 0, stack);
    // LOOPCHECK: Child stack: [[CHILD_STACK:0x[0-9a-f]*]]
    // LOOPCHECK: Main context from: [[CHILD_STACK]] 524288
    ret += Run(argc - 1, 1, stack);
    // LOOPCHECK: Child stack: [[CHILD_STACK:0x[0-9a-f]*]]
    // LOOPCHECK: Main context from: [[CHILD_STACK]] 524288
    ret += Run(argc - 1, 2, stack);
    // LOOPCHECK: Child stack: [[CHILD_STACK:0x[0-9a-f]*]]
    // LOOPCHECK: NextChild stack: [[NEXT_CHILD_STACK:0x[0-9a-f]*]]
    // LOOPCHECK: NextChild from: [[CHILD_STACK]] 524288
    // LOOPCHECK: Main context from: [[NEXT_CHILD_STACK]] 524288
    ret += Run(argc - 1, 0, heap);
    ret += Run(argc - 1, 1, heap);
    ret += Run(argc - 1, 2, heap);
    printf("Iteration %d passed\n", i);
  }

  // CHECK: Test passed
  printf("Test passed\n");

  delete[] heap;
  delete[] next_child_stack;

  return ret;
}
