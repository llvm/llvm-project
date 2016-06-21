// Check that ASan plays well with annotated makecontext/swapcontext.

// RUN: %clangxx_asan -lpthread -O0 %s -o %t && %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -lpthread -O1 %s -o %t && %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -lpthread -O2 %s -o %t && %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -lpthread -O3 %s -o %t && %run %t 2>&1 | FileCheck %s
//
// This test is too subtle to try on non-x86 arch for now.
// REQUIRES: x86_64-supported-target,i386-supported-target

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

void *main_thread_stack;
size_t main_thread_stacksize;

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
  __sanitizer_finish_switch_fiber();

  char x[32] = {0};  // Stack gets poisoned.
  printf("NextChild: %p\n", x);

  CallNoReturn();

  __sanitizer_start_switch_fiber(main_thread_stack, main_thread_stacksize);
  CallNoReturn();
  if (swapcontext(&next_child_context, &orig_context) < 0) {
    perror("swapcontext");
    _exit(1);
  }
}

void Child(int mode) {
  CallNoReturn();
  __sanitizer_finish_switch_fiber();
  char x[32] = {0};  // Stack gets poisoned.
  printf("Child: %p\n", x);
  CallNoReturn();
  // (a) Do nothing, just return to parent function.
  // (b) Jump into the original function. Stack remains poisoned unless we do
  //     something.
  // (c) Jump to another function which will then jump back to the main function
  if (mode == 0) {
    __sanitizer_start_switch_fiber(main_thread_stack, main_thread_stacksize);
    CallNoReturn();
  } else if (mode == 1) {
    __sanitizer_start_switch_fiber(main_thread_stack, main_thread_stacksize);
    CallNoReturn();
    if (swapcontext(&child_context, &orig_context) < 0) {
      perror("swapcontext");
      _exit(1);
    }
  } else if (mode == 2) {
    getcontext(&next_child_context);
    next_child_context.uc_stack.ss_sp = next_child_stack;
    next_child_context.uc_stack.ss_size = kStackSize / 2;
    makecontext(&next_child_context, (void (*)())NextChild, 0);
    __sanitizer_start_switch_fiber(next_child_context.uc_stack.ss_sp,
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
  __sanitizer_start_switch_fiber(child_context.uc_stack.ss_sp,
                                 child_context.uc_stack.ss_size);
  CallNoReturn();
  if (swapcontext(&orig_context, &child_context) < 0) {
    perror("swapcontext");
    _exit(1);
  }
  CallNoReturn();
  __sanitizer_finish_switch_fiber();
  CallNoReturn();

  // Touch childs's stack to make sure it's unpoisoned.
  for (int i = 0; i < kStackSize; i++) {
    child_stack[i] = i;
  }
  return child_stack[arg];
}

void handler(int sig) { CallNoReturn(); }

void InitStackBounds() {
  pthread_attr_t attr;
  pthread_attr_init(&attr);
  pthread_getattr_np(pthread_self(), &attr);
  pthread_attr_getstack(&attr, &main_thread_stack, &main_thread_stacksize);
  pthread_attr_destroy(&attr);
}

int main(int argc, char **argv) {
  InitStackBounds();

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
    ret += Run(argc - 1, 1, stack);
    ret += Run(argc - 1, 2, stack);
    ret += Run(argc - 1, 0, heap);
    ret += Run(argc - 1, 1, heap);
    ret += Run(argc - 1, 2, heap);
  }
  // CHECK: Test passed
  printf("Test passed\n");

  delete[] heap;
  delete[] next_child_stack;

  return ret;
}
