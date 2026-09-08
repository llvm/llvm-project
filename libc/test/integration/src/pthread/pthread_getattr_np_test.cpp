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

static size_t pagesize() {
  return static_cast<size_t>(LIBC_NAMESPACE::sysconf(_SC_PAGESIZE));
}

static void check_readable(const void *start, size_t size) {
  size_t page_size = pagesize();
  auto *bytes = static_cast<const volatile char *>(start);
  for (size_t offset = 0; offset < size; offset += page_size)
    (void)bytes[offset];
  if (size > 0)
    (void)bytes[size - 1];
}

static void wait_barrier(pthread_barrier_t &barrier) {
  int res = LIBC_NAMESPACE::pthread_barrier_wait(&barrier);
  ASSERT_TRUE(res == 0 || res == PTHREAD_BARRIER_SERIAL_THREAD);
}

static pthread_barrier_t ready_barrier;
static pthread_barrier_t done_barrier;

struct PthreadAttrValues {
  int detachstate{-1};
  void *stackaddr{reinterpret_cast<void *>(1)};
  size_t stacksize{1234};
  size_t guardsize{1234};

  PthreadAttrValues() = default;

  void populate_from(pthread_t th) {
    pthread_attr_t attr;
    ASSERT_EQ(LIBC_NAMESPACE::pthread_getattr_np(th, &attr), 0);
    ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_getdetachstate(&attr, &detachstate),
              0);
    ASSERT_EQ(
        LIBC_NAMESPACE::pthread_attr_getstack(&attr, &stackaddr, &stacksize),
        0);
    ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_getguardsize(&attr, &guardsize), 0);
    ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_destroy(&attr), 0);
  }
};

// Test 1: Main thread attributes
// Verifies that pthread_getattr_np on the main thread reports a detached state,
// a dynamic stack size (PTHREAD_STACK_DYNAMIC_NP), and a zero guard size.
static void test_main_thread() {
  PthreadAttrValues values;
  values.populate_from(LIBC_NAMESPACE::pthread_self());

  ASSERT_EQ(values.detachstate, static_cast<int>(PTHREAD_CREATE_DETACHED));
  ASSERT_NE(values.stackaddr, static_cast<void *>(nullptr));
  ASSERT_EQ(values.stacksize, static_cast<size_t>(PTHREAD_STACK_DYNAMIC_NP));
  ASSERT_EQ(reinterpret_cast<uintptr_t>(values.stackaddr) % pagesize(),
            static_cast<uintptr_t>(0));

  uintptr_t local_var_addr = reinterpret_cast<uintptr_t>(&values);
  uintptr_t stack_high = reinterpret_cast<uintptr_t>(values.stackaddr);
  ASSERT_TRUE(local_var_addr < stack_high);
  check_readable(&values, stack_high - local_var_addr);

  ASSERT_EQ(values.guardsize, static_cast<size_t>(0));
}

static void *child_default_func(void *arg) {
  auto *values = static_cast<PthreadAttrValues *>(arg);

  values->populate_from(LIBC_NAMESPACE::pthread_self());

  ASSERT_EQ(values->detachstate, static_cast<int>(PTHREAD_CREATE_JOINABLE));

  ASSERT_NE(values->stackaddr, static_cast<void *>(nullptr));
  ASSERT_NE(values->stacksize, static_cast<size_t>(PTHREAD_STACK_DYNAMIC_NP));
  ASSERT_TRUE(values->stacksize > 0);

  int local_var = 0;
  uintptr_t local_var_addr = reinterpret_cast<uintptr_t>(&local_var);
  uintptr_t stack_low = reinterpret_cast<uintptr_t>(values->stackaddr);
  uintptr_t stack_high = stack_low + values->stacksize;
  ASSERT_TRUE(local_var_addr >= stack_low);
  ASSERT_TRUE(local_var_addr < stack_high);

  check_readable(values->stackaddr, values->stacksize);

  ASSERT_EQ(values->guardsize, pagesize());

  wait_barrier(ready_barrier);
  wait_barrier(done_barrier);

  return nullptr;
}

// Test 2: Child thread with default attributes
// Verifies that a joinable thread created with default attributes reports
// joinable state, an implementation-allocated stack, and a page-sized guard,
// both when queried by the thread itself and by the parent thread.
static void test_child_thread_default() {
  PthreadAttrValues child_values;
  pthread_t th;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_create(&th, nullptr, child_default_func,
                                           &child_values),
            0);

  wait_barrier(ready_barrier);

  // Query from the parent thread while child is still running.
  PthreadAttrValues parent_values;
  parent_values.populate_from(th);

  ASSERT_EQ(parent_values.detachstate,
            static_cast<int>(PTHREAD_CREATE_JOINABLE));
  ASSERT_EQ(parent_values.stackaddr, child_values.stackaddr);
  ASSERT_EQ(parent_values.stacksize, child_values.stacksize);
  ASSERT_EQ(parent_values.guardsize, child_values.guardsize);

  wait_barrier(done_barrier);

  void *retval = nullptr;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_join(th, &retval), 0);
}

static void *child_custom_func(void *arg) {
  auto *values = static_cast<PthreadAttrValues *>(arg);
  values->populate_from(LIBC_NAMESPACE::pthread_self());
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

  PthreadAttrValues values;
  pthread_t th;
  ASSERT_EQ(
      LIBC_NAMESPACE::pthread_create(&th, &attr, child_custom_func, &values),
      0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_destroy(&attr), 0);

  void *retval = nullptr;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_join(th, &retval), 0);

  ASSERT_EQ(values.stackaddr, custom_stack);
  ASSERT_EQ(values.stacksize, custom_stacksize);
  ASSERT_EQ(values.guardsize, static_cast<size_t>(0));
  ASSERT_EQ(values.detachstate, static_cast<int>(PTHREAD_CREATE_JOINABLE));

  ASSERT_EQ(LIBC_NAMESPACE::munmap(custom_stack, custom_stacksize), 0);
}

static void *child_detached_func(void *) {
  PthreadAttrValues values;
  values.populate_from(LIBC_NAMESPACE::pthread_self());

  ASSERT_EQ(values.detachstate, static_cast<int>(PTHREAD_CREATE_DETACHED));
  ASSERT_EQ(values.guardsize, pagesize());

  wait_barrier(done_barrier);
  return nullptr;
}

// Test 4: Child thread created detached
// Verifies that a thread created with PTHREAD_CREATE_DETACHED reports a
// detached state.
static void test_child_thread_detached() {
  pthread_attr_t attr;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_init(&attr), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_setdetachstate(
                &attr, PTHREAD_CREATE_DETACHED),
            0);

  pthread_t th;
  ASSERT_EQ(
      LIBC_NAMESPACE::pthread_create(&th, &attr, child_detached_func, nullptr),
      0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_destroy(&attr), 0);

  wait_barrier(done_barrier);
}

static void *child_dynamic_detach_func(void *) {
  wait_barrier(ready_barrier);
  wait_barrier(done_barrier);
  return nullptr;
}

// Test 5: Dynamically detached child thread
// Verifies that detaching a running joinable thread transitions its reported
// detach state from PTHREAD_CREATE_JOINABLE to PTHREAD_CREATE_DETACHED.
static void test_child_thread_dynamic_detach() {
  pthread_t th;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_create(&th, nullptr,
                                           child_dynamic_detach_func, nullptr),
            0);

  wait_barrier(ready_barrier);

  PthreadAttrValues values;
  values.populate_from(th);
  ASSERT_EQ(values.detachstate, static_cast<int>(PTHREAD_CREATE_JOINABLE));

  ASSERT_EQ(LIBC_NAMESPACE::pthread_detach(th), 0);

  values.populate_from(th);
  ASSERT_EQ(values.detachstate, static_cast<int>(PTHREAD_CREATE_DETACHED));

  wait_barrier(done_barrier);
}

TEST_MAIN() {
  // Barriers cannot be destroyed safely due to issue #221680.
  ASSERT_EQ(LIBC_NAMESPACE::pthread_barrier_init(&ready_barrier, nullptr, 2),
            0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_barrier_init(&done_barrier, nullptr, 2), 0);

  test_main_thread();
  test_child_thread_default();
  test_child_thread_custom_stack();
  test_child_thread_detached();
  test_child_thread_dynamic_detach();
  return 0;
}
