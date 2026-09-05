//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Integration tests for pthread_getattr_np.
///
//===----------------------------------------------------------------------===//

#include "hdr/pthread_macros.h"
#include "hdr/stdint_proxy.h"
#include "hdr/sys_mman_macros.h"
#include "src/pthread/pthread_attr_destroy.h"
#include "src/pthread/pthread_attr_getdetachstate.h"
#include "src/pthread/pthread_attr_getguardsize.h"
#include "src/pthread/pthread_attr_getstack.h"
#include "src/pthread/pthread_attr_init.h"
#include "src/pthread/pthread_attr_setdetachstate.h"
#include "src/pthread/pthread_attr_setstack.h"
#include "src/pthread/pthread_barrier_destroy.h"
#include "src/pthread/pthread_barrier_init.h"
#include "src/pthread/pthread_barrier_wait.h"
#include "src/pthread/pthread_create.h"
#include "src/pthread/pthread_detach.h"
#include "src/pthread/pthread_getattr_np.h"
#include "src/pthread/pthread_join.h"
#include "src/pthread/pthread_self.h"
#include "src/sys/mman/mmap.h"
#include "src/sys/mman/munmap.h"
#include "src/unistd/sysconf.h"
#include "test/IntegrationTest/test.h"

static void check_readable(const void *start, size_t size) {
  size_t pagesize = LIBC_NAMESPACE::sysconf(_SC_PAGESIZE);
  auto *bytes = static_cast<const volatile char *>(start);
  for (size_t offset = 0; offset < size; offset += pagesize)
    (void)bytes[offset];
  if (size > 0)
    (void)bytes[size - 1];
}

static void wait_barrier(pthread_barrier_t &barrier) {
  int res = LIBC_NAMESPACE::pthread_barrier_wait(&barrier);
  ASSERT_TRUE(res == 0 || res == PTHREAD_BARRIER_SERIAL_THREAD);
}

// Test 1: Main thread attributes
// Verifies that pthread_getattr_np on the main thread reports a detached state,
// a dynamic stack size (PTHREAD_STACK_DYNAMIC_NP), and a zero guard size.
static void test_main_thread() {
  pthread_attr_t attr;
  ASSERT_EQ(
      LIBC_NAMESPACE::pthread_getattr_np(LIBC_NAMESPACE::pthread_self(), &attr),
      0);

  int detachstate = -1;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_getdetachstate(&attr, &detachstate),
            0);
  ASSERT_EQ(detachstate, static_cast<int>(PTHREAD_CREATE_DETACHED));

  void *stackaddr = nullptr;
  size_t stacksize = 1234;
  ASSERT_EQ(
      LIBC_NAMESPACE::pthread_attr_getstack(&attr, &stackaddr, &stacksize), 0);
  ASSERT_NE(stackaddr, static_cast<void *>(nullptr));
  ASSERT_EQ(stacksize, static_cast<size_t>(PTHREAD_STACK_DYNAMIC_NP));
  ASSERT_EQ(reinterpret_cast<uintptr_t>(stackaddr) %
                LIBC_NAMESPACE::sysconf(_SC_PAGESIZE),
            static_cast<uintptr_t>(0));

  uintptr_t local_var_addr = reinterpret_cast<uintptr_t>(&attr);
  ASSERT_TRUE(local_var_addr < reinterpret_cast<uintptr_t>(stackaddr));
  check_readable(&attr,
                 reinterpret_cast<uintptr_t>(stackaddr) - local_var_addr);

  size_t guardsize = 1234;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_getguardsize(&attr, &guardsize), 0);
  ASSERT_EQ(guardsize, static_cast<size_t>(0));

  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_destroy(&attr), 0);
}

struct ChildDefaultArgs {
  pthread_barrier_t ready_barrier;
  pthread_barrier_t done_barrier;
  void *stackaddr{nullptr};
  size_t stacksize{0};
  size_t guardsize{0};
  int detachstate{-1};
};

static void *child_default_func(void *arg) {
  auto *args = static_cast<ChildDefaultArgs *>(arg);

  pthread_attr_t attr;
  ASSERT_EQ(
      LIBC_NAMESPACE::pthread_getattr_np(LIBC_NAMESPACE::pthread_self(), &attr),
      0);

  ASSERT_EQ(
      LIBC_NAMESPACE::pthread_attr_getdetachstate(&attr, &args->detachstate),
      0);
  ASSERT_EQ(args->detachstate, static_cast<int>(PTHREAD_CREATE_JOINABLE));

  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_getstack(&attr, &args->stackaddr,
                                                  &args->stacksize),
            0);
  ASSERT_NE(args->stackaddr, static_cast<void *>(nullptr));
  ASSERT_NE(args->stacksize, static_cast<size_t>(PTHREAD_STACK_DYNAMIC_NP));
  ASSERT_TRUE(args->stacksize > 0);

  uintptr_t local_var_addr = reinterpret_cast<uintptr_t>(&attr);
  uintptr_t stack_low = reinterpret_cast<uintptr_t>(args->stackaddr);
  uintptr_t stack_high = stack_low + args->stacksize;
  ASSERT_TRUE(local_var_addr >= stack_low);
  ASSERT_TRUE(local_var_addr < stack_high);

  check_readable(args->stackaddr, args->stacksize);

  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_getguardsize(&attr, &args->guardsize),
            0);
  ASSERT_EQ(args->guardsize,
            static_cast<size_t>(LIBC_NAMESPACE::sysconf(_SC_PAGESIZE)));

  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_destroy(&attr), 0);

  wait_barrier(args->ready_barrier);
  wait_barrier(args->done_barrier);

  return nullptr;
}

// Test 2: Child thread with default attributes
// Verifies that a joinable thread created with default attributes reports
// joinable state, an implementation-allocated stack, and a page-sized guard,
// both when queried by the thread itself and by the parent thread.
static void test_child_thread_default() {
  ChildDefaultArgs args;
  ASSERT_EQ(
      LIBC_NAMESPACE::pthread_barrier_init(&args.ready_barrier, nullptr, 2), 0);
  ASSERT_EQ(
      LIBC_NAMESPACE::pthread_barrier_init(&args.done_barrier, nullptr, 2), 0);

  pthread_t th;
  ASSERT_EQ(
      LIBC_NAMESPACE::pthread_create(&th, nullptr, child_default_func, &args),
      0);

  wait_barrier(args.ready_barrier);

  // Query from the parent thread while child is still running.
  pthread_attr_t attr;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_getattr_np(th, &attr), 0);

  int detachstate = -1;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_getdetachstate(&attr, &detachstate),
            0);
  ASSERT_EQ(detachstate, static_cast<int>(PTHREAD_CREATE_JOINABLE));

  void *stackaddr = nullptr;
  size_t stacksize = 0;
  ASSERT_EQ(
      LIBC_NAMESPACE::pthread_attr_getstack(&attr, &stackaddr, &stacksize), 0);
  ASSERT_EQ(stackaddr, args.stackaddr);
  ASSERT_EQ(stacksize, args.stacksize);

  size_t guardsize = 0;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_getguardsize(&attr, &guardsize), 0);
  ASSERT_EQ(guardsize, args.guardsize);

  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_destroy(&attr), 0);

  wait_barrier(args.done_barrier);

  void *retval = nullptr;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_join(th, &retval), 0);

  ASSERT_EQ(LIBC_NAMESPACE::pthread_barrier_destroy(&args.ready_barrier), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_barrier_destroy(&args.done_barrier), 0);
}

struct ChildCustomArgs {
  void *stackaddr{nullptr};
  size_t stacksize{0};
  size_t guardsize{0};
  int detachstate{-1};
};

static void *child_custom_func(void *arg) {
  auto *args = static_cast<ChildCustomArgs *>(arg);

  pthread_attr_t attr;
  ASSERT_EQ(
      LIBC_NAMESPACE::pthread_getattr_np(LIBC_NAMESPACE::pthread_self(), &attr),
      0);

  ASSERT_EQ(
      LIBC_NAMESPACE::pthread_attr_getdetachstate(&attr, &args->detachstate),
      0);
  ASSERT_EQ(args->detachstate, static_cast<int>(PTHREAD_CREATE_JOINABLE));

  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_getstack(&attr, &args->stackaddr,
                                                  &args->stacksize),
            0);

  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_getguardsize(&attr, &args->guardsize),
            0);
  ASSERT_EQ(args->guardsize, static_cast<size_t>(0));

  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_destroy(&attr), 0);

  return nullptr;
}

// Test 3: Child thread with custom stack
// Verifies that a thread created with a user-allocated stack reports the exact
// stack address and size, and reports a guard size of 0.
static void test_child_thread_custom_stack() {
  size_t custom_stacksize = PTHREAD_STACK_MIN * 2;
  void *custom_stack =
      LIBC_NAMESPACE::mmap(nullptr, custom_stacksize, PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  ASSERT_NE(custom_stack, MAP_FAILED);
  ASSERT_NE(custom_stack, static_cast<void *>(nullptr));

  pthread_attr_t attr;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_init(&attr), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_setstack(&attr, custom_stack,
                                                  custom_stacksize),
            0);

  ChildCustomArgs args;
  pthread_t th;
  ASSERT_EQ(
      LIBC_NAMESPACE::pthread_create(&th, &attr, child_custom_func, &args), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_destroy(&attr), 0);

  void *retval = nullptr;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_join(th, &retval), 0);

  ASSERT_EQ(args.stackaddr, custom_stack);
  ASSERT_EQ(args.stacksize, custom_stacksize);
  ASSERT_EQ(args.guardsize, static_cast<size_t>(0));
  ASSERT_EQ(args.detachstate, static_cast<int>(PTHREAD_CREATE_JOINABLE));

  ASSERT_EQ(LIBC_NAMESPACE::munmap(custom_stack, custom_stacksize), 0);
}

static void *child_detached_func(void *arg) {
  auto &barrier = *static_cast<pthread_barrier_t *>(arg);

  pthread_attr_t attr;
  ASSERT_EQ(
      LIBC_NAMESPACE::pthread_getattr_np(LIBC_NAMESPACE::pthread_self(), &attr),
      0);

  int detachstate = -1;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_getdetachstate(&attr, &detachstate),
            0);
  ASSERT_EQ(detachstate, static_cast<int>(PTHREAD_CREATE_DETACHED));

  size_t guardsize = 0;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_getguardsize(&attr, &guardsize), 0);
  ASSERT_EQ(guardsize,
            static_cast<size_t>(LIBC_NAMESPACE::sysconf(_SC_PAGESIZE)));

  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_destroy(&attr), 0);

  wait_barrier(barrier);
  return nullptr;
}

// Test 4: Child thread created detached
// Verifies that a thread created with PTHREAD_CREATE_DETACHED reports a
// detached state.
static void test_child_thread_detached() {
  pthread_barrier_t barrier;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_barrier_init(&barrier, nullptr, 2), 0);

  pthread_attr_t attr;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_init(&attr), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_setdetachstate(
                &attr, PTHREAD_CREATE_DETACHED),
            0);

  pthread_t th;
  ASSERT_EQ(
      LIBC_NAMESPACE::pthread_create(&th, &attr, child_detached_func, &barrier),
      0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_destroy(&attr), 0);

  wait_barrier(barrier);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_barrier_destroy(&barrier), 0);
}

struct DynamicDetachArgs {
  pthread_barrier_t ready_barrier;
  pthread_barrier_t done_barrier;
};

static void *child_dynamic_detach_func(void *arg) {
  auto *args = static_cast<DynamicDetachArgs *>(arg);
  wait_barrier(args->ready_barrier);
  wait_barrier(args->done_barrier);
  return nullptr;
}

// Test 5: Dynamically detached child thread
// Verifies that detaching a running joinable thread transitions its reported
// detach state from PTHREAD_CREATE_JOINABLE to PTHREAD_CREATE_DETACHED.
static void test_child_thread_dynamic_detach() {
  DynamicDetachArgs args;
  ASSERT_EQ(
      LIBC_NAMESPACE::pthread_barrier_init(&args.ready_barrier, nullptr, 2), 0);
  ASSERT_EQ(
      LIBC_NAMESPACE::pthread_barrier_init(&args.done_barrier, nullptr, 2), 0);

  pthread_t th;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_create(&th, nullptr,
                                           child_dynamic_detach_func, &args),
            0);

  wait_barrier(args.ready_barrier);

  pthread_attr_t attr;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_getattr_np(th, &attr), 0);
  int detachstate = -1;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_getdetachstate(&attr, &detachstate),
            0);
  ASSERT_EQ(detachstate, static_cast<int>(PTHREAD_CREATE_JOINABLE));
  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_destroy(&attr), 0);

  ASSERT_EQ(LIBC_NAMESPACE::pthread_detach(th), 0);

  ASSERT_EQ(LIBC_NAMESPACE::pthread_getattr_np(th, &attr), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_getdetachstate(&attr, &detachstate),
            0);
  ASSERT_EQ(detachstate, static_cast<int>(PTHREAD_CREATE_DETACHED));
  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_destroy(&attr), 0);

  wait_barrier(args.done_barrier);

  ASSERT_EQ(LIBC_NAMESPACE::pthread_barrier_destroy(&args.ready_barrier), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_barrier_destroy(&args.done_barrier), 0);
}

TEST_MAIN() {
  test_main_thread();
  test_child_thread_default();
  test_child_thread_custom_stack();
  test_child_thread_detached();
  test_child_thread_dynamic_detach();
  return 0;
}
