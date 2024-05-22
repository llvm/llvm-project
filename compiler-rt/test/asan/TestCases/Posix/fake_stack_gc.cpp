// RUN: %clangxx_asan -O0 -pthread %s -o %t && %env_asan_opts=use_sigaltstack=0 %run not --crash %t 2>&1 | FileCheck %s

// Check that fake stack does not discard frames on the main stack, when GC is
// triggered from high alt stack.

// This test does not work on iOS simulator
// (https://github.com/llvm/llvm-project/issues/64942).
// UNSUPPORTED: iossim

#include <algorithm>
#include <assert.h>
#include <csignal>
#include <cstdint>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

const size_t kStackSize = 0x100000;

int *on_thread;
int *p;

template <size_t N> void Fn() {
  int t[N];
  p = t;
  if constexpr (N > 1)
    Fn<N - 1>();
}

static void Handler(int signo) {
  fprintf(stderr, "Handler Frame:%p\n", __builtin_frame_address(0));

  // Trigger GC and create a lot of frame to reuse "Thread" frame if it was
  // discarded.
  for (int i = 0; i < 1000; ++i)
    Fn<100>();
  // If we discarder and reused "Thread" frame, the next line will crash with
  // false report.
  *on_thread = 10;
  fprintf(stderr, "SUCCESS\n");
  // CHECK: SUCCESS
}

void *Thread(void *arg) {
  fprintf(stderr, "Thread Frame:%p\n", __builtin_frame_address(0));
  stack_t stack = {};
  stack.ss_sp = arg;
  stack.ss_flags = 0;
  stack.ss_size = kStackSize;
  assert(sigaltstack(&stack, nullptr) == 0);

  struct sigaction sa = {};
  sa.sa_handler = Handler;
  sa.sa_flags = SA_ONSTACK;
  sigaction(SIGABRT, &sa, nullptr);

  // Store pointer to the local var, so we can access this frame from the signal
  // handler when the frame is still alive.
  int n;
  on_thread = &n;

  // Abort should schedule FakeStack GC and call handler on alt stack.
  abort();
}

int main(void) {
  // Allocate main and alt stack for future thread.
  void *main_stack;
  void *alt_stack;
  size_t const kPageSize = sysconf(_SC_PAGESIZE);
  assert(posix_memalign(&main_stack, kPageSize, kStackSize) == 0);
  assert(posix_memalign(&alt_stack, kPageSize, kStackSize) == 0);

  // Pick the lower stack as the main stack, as we want to trigger GC in
  // FakeStack from alt stack in a such way that main stack is allocated below.
  if ((uintptr_t)main_stack > (uintptr_t)alt_stack)
    std::swap(alt_stack, main_stack);

  fprintf(stderr, "main_stack: %p-%p\n", main_stack,
          (char *)main_stack + kStackSize);
  fprintf(stderr, "alt_stack: %p-%p\n", alt_stack,
          (char *)alt_stack + kStackSize);

  pthread_attr_t attr;
  assert(pthread_attr_init(&attr) == 0);
  assert(pthread_attr_setstack(&attr, main_stack, kStackSize) == 0);

  pthread_t tid;
  assert(pthread_create(&tid, &attr, Thread, alt_stack) == 0);

  pthread_join(tid, nullptr);

  free(main_stack);
  free(alt_stack);

  return 0;
}
