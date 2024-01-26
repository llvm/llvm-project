//===-- Tests for pthread_create ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/pthread/pthread_attr_destroy.h"
#include "src/pthread/pthread_attr_getdetachstate.h"
#include "src/pthread/pthread_attr_getguardsize.h"
#include "src/pthread/pthread_attr_getstack.h"
#include "src/pthread/pthread_attr_getstacksize.h"
#include "src/pthread/pthread_attr_init.h"
#include "src/pthread/pthread_attr_setdetachstate.h"
#include "src/pthread/pthread_attr_setguardsize.h"
#include "src/pthread/pthread_attr_setstack.h"
#include "src/pthread/pthread_attr_setstacksize.h"
#include "src/pthread/pthread_create.h"
#include "src/pthread/pthread_join.h"
#include "src/pthread/pthread_self.h"

#include "src/sys/mman/mmap.h"
#include "src/sys/mman/munmap.h"
#include "src/sys/random/getrandom.h"

#include "src/__support/CPP/array.h"
#include "src/__support/CPP/atomic.h"
#include "src/__support/CPP/new.h"
#include "src/__support/threads/thread.h"

#include "src/errno/libc_errno.h"

#include "test/IntegrationTest/test.h"

#include <linux/param.h> // For EXEC_PAGESIZE.
#include <pthread.h>

struct TestThreadArgs {
  pthread_attr_t attrs;
  void *ret;
};
static LIBC_NAMESPACE::AllocChecker global_ac;
static LIBC_NAMESPACE::cpp::Atomic<long> global_thr_count = 0;

static void *successThread(void *Arg) {
  pthread_t th = LIBC_NAMESPACE::pthread_self();
  auto *thread = reinterpret_cast<LIBC_NAMESPACE::Thread *>(&th);

  ASSERT_ERRNO_SUCCESS();
  ASSERT_TRUE(thread);
  ASSERT_TRUE(thread->attrib);

  TestThreadArgs *th_arg = reinterpret_cast<TestThreadArgs *>(Arg);
  pthread_attr_t *expec_attrs = &(th_arg->attrs);
  void *ret = th_arg->ret;

  void *expec_stack;
  size_t expec_stacksize, expec_guardsize, expec_stacksize2;
  int expec_detached;

  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_getstack(expec_attrs, &expec_stack,
                                                  &expec_stacksize),
            0);
  ASSERT_ERRNO_SUCCESS();

  ASSERT_EQ(
      LIBC_NAMESPACE::pthread_attr_getstacksize(expec_attrs, &expec_stacksize2),
      0);
  ASSERT_ERRNO_SUCCESS();

  ASSERT_EQ(
      LIBC_NAMESPACE::pthread_attr_getguardsize(expec_attrs, &expec_guardsize),
      0);
  ASSERT_ERRNO_SUCCESS();

  ASSERT_EQ(
      LIBC_NAMESPACE::pthread_attr_getdetachstate(expec_attrs, &expec_detached),
      0);
  ASSERT_ERRNO_SUCCESS();

  ASSERT_EQ(expec_stacksize, expec_stacksize2);

  ASSERT_TRUE(thread->attrib->stack);
  if (expec_stack != nullptr) {
    ASSERT_EQ(thread->attrib->stack, expec_stack);
  } else {
    ASSERT_EQ(reinterpret_cast<uintptr_t>(thread->attrib->stack) %
                  EXEC_PAGESIZE,
              static_cast<uintptr_t>(0));
    expec_stacksize = (expec_stacksize + EXEC_PAGESIZE - 1) & (-EXEC_PAGESIZE);
  }

  ASSERT_TRUE(expec_stacksize);
  ASSERT_EQ(thread->attrib->stacksize, expec_stacksize);
  ASSERT_EQ(thread->attrib->guardsize, expec_guardsize);

  ASSERT_EQ(expec_detached == PTHREAD_CREATE_JOINABLE,
            thread->attrib->detach_state.load() ==
                static_cast<uint32_t>(LIBC_NAMESPACE::DetachState::JOINABLE));
  ASSERT_EQ(expec_detached == PTHREAD_CREATE_DETACHED,
            thread->attrib->detach_state.load() ==
                static_cast<uint32_t>(LIBC_NAMESPACE::DetachState::DETACHED));

  {
    // Allocate some bytes on the stack on most of the stack and make sure we
    // have read/write permissions on the memory.
    size_t test_stacksize = expec_stacksize - 1024;
    volatile uint8_t *bytes_on_stack =
        (volatile uint8_t *)__builtin_alloca(test_stacksize);

    for (size_t I = 0; I < test_stacksize; ++I) {
      // Write permissions
      bytes_on_stack[I] = static_cast<uint8_t>(I);
    }

    for (size_t I = 0; I < test_stacksize; ++I) {
      // Read/write permissions
      bytes_on_stack[I] += static_cast<uint8_t>(I);
    }
  }

  // TODO: If guardsize != 0 && expec_stack == nullptr we should confirm that
  // [stack - expec_guardsize, stack) is both mapped and has PROT_NONE
  // permissions. Maybe we can read from /proc/{self}/map?

  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_destroy(expec_attrs), 0);
  ASSERT_ERRNO_SUCCESS();

  // Arg is malloced, so free.
  delete th_arg;
  global_thr_count.fetch_sub(1);
  return ret;
}

static void run_success_config(int detachstate, size_t guardsize,
                               size_t stacksize, bool customstack) {

  TestThreadArgs *th_arg = new (global_ac) TestThreadArgs{};
  pthread_attr_t *attr = &(th_arg->attrs);

  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_init(attr), 0);
  ASSERT_ERRNO_SUCCESS();

  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_setdetachstate(attr, detachstate), 0);
  ASSERT_ERRNO_SUCCESS();

  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_setguardsize(attr, guardsize), 0);
  ASSERT_ERRNO_SUCCESS();

  void *Stack = nullptr;
  if (customstack) {
    Stack = LIBC_NAMESPACE::mmap(nullptr, stacksize, PROT_READ | PROT_WRITE,
                                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    ASSERT_NE(Stack, MAP_FAILED);
    ASSERT_NE(Stack, static_cast<void *>(nullptr));
    ASSERT_ERRNO_SUCCESS();

    ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_setstack(attr, Stack, stacksize), 0);
    ASSERT_ERRNO_SUCCESS();
  } else {
    ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_setstacksize(attr, stacksize), 0);
    ASSERT_ERRNO_SUCCESS();
  }

  void *expec_ret = nullptr;
  if (detachstate == PTHREAD_CREATE_JOINABLE) {
    ASSERT_EQ(LIBC_NAMESPACE::getrandom(&expec_ret, sizeof(expec_ret), 0),
              static_cast<ssize_t>(sizeof(expec_ret)));
    ASSERT_ERRNO_SUCCESS();
  }

  th_arg->ret = expec_ret;
  global_thr_count.fetch_add(1);

  pthread_t tid;
  // th_arg and attr are cleanup by the thread.
  ASSERT_EQ(LIBC_NAMESPACE::pthread_create(&tid, attr, successThread,
                                           reinterpret_cast<void *>(th_arg)),
            0);
  ASSERT_ERRNO_SUCCESS();

  if (detachstate == PTHREAD_CREATE_JOINABLE) {
    void *th_ret;
    ASSERT_EQ(LIBC_NAMESPACE::pthread_join(tid, &th_ret), 0);
    ASSERT_ERRNO_SUCCESS();
    ASSERT_EQ(th_ret, expec_ret);

    if (customstack) {
      ASSERT_EQ(LIBC_NAMESPACE::munmap(Stack, stacksize), 0);
      ASSERT_ERRNO_SUCCESS();
    }
  } else {
    ASSERT_FALSE(customstack);
  }
}

static void run_success_tests() {

  // Test parameters
  using LIBC_NAMESPACE::cpp::array;

  array<int, 2> detachstates = {PTHREAD_CREATE_DETACHED,
                                PTHREAD_CREATE_JOINABLE};
  array<size_t, 4> guardsizes = {0, EXEC_PAGESIZE, 2 * EXEC_PAGESIZE,
                                 123 * EXEC_PAGESIZE};
  array<size_t, 6> stacksizes = {PTHREAD_STACK_MIN,
                                 PTHREAD_STACK_MIN + 16,
                                 (1 << 16) - EXEC_PAGESIZE / 2,
                                 (1 << 16) + EXEC_PAGESIZE / 2,
                                 1234560,
                                 1234560 * 2};
  array<bool, 2> customstacks = {true, false};

  for (int detachstate : detachstates) {
    for (size_t guardsize : guardsizes) {
      for (size_t stacksize : stacksizes) {
        for (bool customstack : customstacks) {
          if (customstack) {

            // TODO: figure out how to test a user allocated stack
            // along with detached pthread safely. We can't let the
            // thread deallocate it owns stack for obvious
            // reasons. And there doesn't appear to be a good way to
            // check if a detached thread has exited. NB: It's racey to just
            // wait for an atomic variable at the end of the thread function as
            // internal thread cleanup functions continue to use its stack.
            // Maybe an `atexit` handler would work.
            if (detachstate == PTHREAD_CREATE_DETACHED)
              continue;

            // Guardsize has no meaning with user provided stack.
            if (guardsize)
              continue;

            run_success_config(detachstate, guardsize, stacksize, customstack);
          }
        }
      }
    }
  }

  // Wait for detached threads to finish testing (this is not gurantee they will
  // have cleaned up)
  while (global_thr_count.load())
    ;
}

static void *failure_thread(void *) {
  // Should be unreachable;
  ASSERT_TRUE(false);
  return nullptr;
}

static void create_and_check_failure_thread(pthread_attr_t *attr) {
  pthread_t tid;
  int result =
      LIBC_NAMESPACE::pthread_create(&tid, attr, failure_thread, nullptr);
  // EINVAL if we caught on overflow or something of that nature. EAGAIN if it
  // was just really larger we failed mmap.
  ASSERT_TRUE(result == EINVAL || result == EAGAIN);
  // pthread_create should NOT set errno on error
  ASSERT_ERRNO_SUCCESS();

  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_destroy(attr), 0);
  ASSERT_ERRNO_SUCCESS();
}

static void run_failure_config(size_t guardsize, size_t stacksize) {
  pthread_attr_t attr;
  guardsize &= -EXEC_PAGESIZE;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_init(&attr), 0);
  ASSERT_ERRNO_SUCCESS();

  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_setguardsize(&attr, guardsize), 0);
  ASSERT_ERRNO_SUCCESS();

  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_setstacksize(&attr, stacksize), 0);
  ASSERT_ERRNO_SUCCESS();

  create_and_check_failure_thread(&attr);
}

static void run_failure_tests() {
  // Just some tests where the user sets "valid" parameters but they fail
  // (overflow or too large to allocate).
  run_failure_config(SIZE_MAX, PTHREAD_STACK_MIN);
  run_failure_config(SIZE_MAX - PTHREAD_STACK_MIN, PTHREAD_STACK_MIN * 2);
  run_failure_config(PTHREAD_STACK_MIN, SIZE_MAX);
  run_failure_config(PTHREAD_STACK_MIN, SIZE_MAX - PTHREAD_STACK_MIN);
  run_failure_config(SIZE_MAX / 2, SIZE_MAX / 2);
  run_failure_config(3 * (SIZE_MAX / 4), SIZE_MAX / 4);
  run_failure_config(SIZE_MAX / 2 + 1234, SIZE_MAX / 2);

  // Test invalid parameters that are impossible to obtain via the
  // `pthread_attr_set*` API. Still test that this not entirely unlikely
  // initialization doesn't cause any issues. Basically we wan't to make sure
  // that `pthread_create` properly checks for input validity and doesn't rely
  // on the `pthread_attr_set*` API.
  pthread_attr_t attr;

  // Stacksize too small.
  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_init(&attr), 0);
  ASSERT_ERRNO_SUCCESS();
  attr.__stacksize = PTHREAD_STACK_MIN - 16;
  create_and_check_failure_thread(&attr);

  // Stack misaligned.
  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_init(&attr), 0);
  ASSERT_ERRNO_SUCCESS();
  attr.__stack = reinterpret_cast<void *>(1);
  create_and_check_failure_thread(&attr);

  // Stack + stacksize misaligned.
  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_init(&attr), 0);
  ASSERT_ERRNO_SUCCESS();
  attr.__stacksize = PTHREAD_STACK_MIN + 1;
  attr.__stack = reinterpret_cast<void *>(16);
  create_and_check_failure_thread(&attr);

  // Guardsize misaligned.
  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_init(&attr), 0);
  ASSERT_ERRNO_SUCCESS();
  attr.__guardsize = EXEC_PAGESIZE / 2;
  create_and_check_failure_thread(&attr);

  // Detachstate is unknown.
  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_init(&attr), 0);
  ASSERT_ERRNO_SUCCESS();
  attr.__detachstate = -1;
  create_and_check_failure_thread(&attr);
}

TEST_MAIN() {
  libc_errno = 0;
  run_success_tests();
  run_failure_tests();
  return 0;
}
