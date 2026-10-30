//===-- Tests for pthread_create ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/pthread_macros.h"
#include "hdr/sched_macros.h"
#include "hdr/sys_mman_macros.h"
#include "hdr/types/struct_sched_param.h"
#include "src/pthread/pthread_attr_destroy.h"
#include "src/pthread/pthread_attr_getdetachstate.h"
#include "src/pthread/pthread_attr_getguardsize.h"
#include "src/pthread/pthread_attr_getstack.h"
#include "src/pthread/pthread_attr_getstacksize.h"
#include "src/pthread/pthread_attr_init.h"
#include "src/pthread/pthread_attr_setdetachstate.h"
#include "src/pthread/pthread_attr_setguardsize.h"
#include "src/pthread/pthread_attr_setinheritsched.h"
#include "src/pthread/pthread_attr_setschedparam.h"
#include "src/pthread/pthread_attr_setschedpolicy.h"
#include "src/pthread/pthread_attr_setstack.h"
#include "src/pthread/pthread_attr_setstacksize.h"
#include "src/pthread/pthread_create.h"
#include "src/pthread/pthread_getschedparam.h"
#include "src/pthread/pthread_getunique_np.h"
#include "src/pthread/pthread_join.h"
#include "src/pthread/pthread_self.h"

#include "src/sys/mman/mmap.h"
#include "src/sys/mman/munmap.h"
#include "src/sys/random/getrandom.h"

#include "src/__support/CPP/array.h"
#include "src/__support/CPP/atomic.h"
#include "src/__support/CPP/new.h"
#include "src/__support/alloc-checker.h"
#include "src/__support/threads/thread.h"

#include "test/IntegrationTest/test.h"

#include <errno.h>
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

    for (size_t i = 0; i < test_stacksize; ++i) {
      // Write permissions
      bytes_on_stack[i] = static_cast<uint8_t>(i);
    }

    for (size_t i = 0; i < test_stacksize; ++i) {
      // Read/write permissions
      bytes_on_stack[i] += static_cast<uint8_t>(i);
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
  pthread_id_np_t id;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_getunique_np(&tid, &id), 0);
  ASSERT_NE(id, 0);

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

  // Inheritsched is unknown.
  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_init(&attr), 0);
  ASSERT_ERRNO_SUCCESS();
  attr.__inheritsched = -1;
  create_and_check_failure_thread(&attr);

  // Schedpolicy is invalid when explicit sched is requested.
  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_init(&attr), 0);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_setinheritsched(
                &attr, PTHREAD_EXPLICIT_SCHED),
            0);
  ASSERT_ERRNO_SUCCESS();
  attr.__schedpolicy = -1;
  create_and_check_failure_thread(&attr);

  // Sched priority is invalid for policy.
  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_init(&attr), 0);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_setinheritsched(
                &attr, PTHREAD_EXPLICIT_SCHED),
            0);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_setschedpolicy(&attr, SCHED_OTHER), 0);
  ASSERT_ERRNO_SUCCESS();
  sched_param bad_param;
  bad_param.sched_priority = 1;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_setschedparam(&attr, &bad_param), 0);
  ASSERT_ERRNO_SUCCESS();
  create_and_check_failure_thread(&attr);
}

struct SchedThreadArgs {
  LIBC_NAMESPACE::cpp::Atomic<bool> executed = false;
  int policy = 0;
  int priority = 0;
};

static void *sched_runner(void *arg) {
  auto *args = reinterpret_cast<SchedThreadArgs *>(arg);
  sched_param param;
  int policy;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_getschedparam(
                LIBC_NAMESPACE::pthread_self(), &policy, &param),
            0);
  ASSERT_ERRNO_SUCCESS();
  args->policy = policy;
  args->priority = param.sched_priority;
  args->executed.store(true);
  return nullptr;
}

static void test_sched_inherit() {
  pthread_t self = LIBC_NAMESPACE::pthread_self();
  int parent_policy = 0;
  sched_param parent_param;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_getschedparam(self, &parent_policy,
                                                  &parent_param),
            0);
  ASSERT_ERRNO_SUCCESS();

  // 1. With default attr (nullptr)
  {
    SchedThreadArgs args;
    pthread_t tid;
    ASSERT_EQ(
        LIBC_NAMESPACE::pthread_create(&tid, nullptr, sched_runner, &args), 0);
    ASSERT_ERRNO_SUCCESS();
    ASSERT_EQ(LIBC_NAMESPACE::pthread_join(tid, nullptr), 0);
    ASSERT_ERRNO_SUCCESS();
    ASSERT_TRUE(args.executed.load());
    ASSERT_EQ(args.policy, parent_policy);
    ASSERT_EQ(args.priority, parent_param.sched_priority);
  }

  // 2. Explicit PTHREAD_INHERIT_SCHED in attr, while setting schedpolicy to
  // SCHED_BATCH
  {
    pthread_attr_t attr;
    ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_init(&attr), 0);
    ASSERT_ERRNO_SUCCESS();
    ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_setinheritsched(
                  &attr, PTHREAD_INHERIT_SCHED),
              0);
    ASSERT_ERRNO_SUCCESS();
    // Even if schedpolicy is modified in attr, PTHREAD_INHERIT_SCHED means it
    // must be ignored.
    int dummy_policy =
        (parent_policy == SCHED_OTHER) ? SCHED_BATCH : SCHED_OTHER;
    ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_setschedpolicy(&attr, dummy_policy),
              0);
    ASSERT_ERRNO_SUCCESS();

    SchedThreadArgs args;
    pthread_t tid;
    ASSERT_EQ(LIBC_NAMESPACE::pthread_create(&tid, &attr, sched_runner, &args),
              0);
    ASSERT_ERRNO_SUCCESS();
    ASSERT_EQ(LIBC_NAMESPACE::pthread_join(tid, nullptr), 0);
    ASSERT_ERRNO_SUCCESS();
    ASSERT_TRUE(args.executed.load());
    ASSERT_EQ(args.policy, parent_policy);
    ASSERT_EQ(args.priority, parent_param.sched_priority);

    ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_destroy(&attr), 0);
    ASSERT_ERRNO_SUCCESS();
  }
}

static void verify_sched_policy(pthread_attr_t &attr, int expected_policy,
                                int expected_priority) {
  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_setschedpolicy(&attr, expected_policy),
            0);
  ASSERT_ERRNO_SUCCESS();
  sched_param param;
  param.sched_priority = expected_priority;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_setschedparam(&attr, &param), 0);
  ASSERT_ERRNO_SUCCESS();

  SchedThreadArgs args;
  pthread_t tid;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_create(&tid, &attr, sched_runner, &args),
            0);
  ASSERT_ERRNO_SUCCESS();

  ASSERT_EQ(LIBC_NAMESPACE::pthread_join(tid, nullptr), 0);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_TRUE(args.executed.load());
  ASSERT_EQ(args.policy, expected_policy);
  ASSERT_EQ(args.priority, expected_priority);
}

static void test_sched_explicit_success() {
  pthread_attr_t attr;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_init(&attr), 0);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_setinheritsched(
                &attr, PTHREAD_EXPLICIT_SCHED),
            0);
  ASSERT_ERRNO_SUCCESS();

  verify_sched_policy(attr, SCHED_OTHER, 0);
  verify_sched_policy(attr, SCHED_BATCH, 0);

  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_destroy(&attr), 0);
  ASSERT_ERRNO_SUCCESS();
}

static void test_sched_realtime_permission() {
  pthread_attr_t attr;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_init(&attr), 0);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_setinheritsched(
                &attr, PTHREAD_EXPLICIT_SCHED),
            0);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_setschedpolicy(&attr, SCHED_FIFO), 0);
  ASSERT_ERRNO_SUCCESS();
  sched_param param;
  param.sched_priority = 1;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_setschedparam(&attr, &param), 0);
  ASSERT_ERRNO_SUCCESS();

  SchedThreadArgs args;
  pthread_t tid;
  int res = LIBC_NAMESPACE::pthread_create(&tid, &attr, sched_runner, &args);
  // Setting realtime scheduling policy SCHED_FIFO requires CAP_SYS_NICE or
  // sufficient RLIMIT_RTPRIO. In unprivileged environments this will fail with
  // EPERM, while in privileged environments it will succeed with 0.
  ASSERT_TRUE(res == EPERM || res == 0);
  ASSERT_ERRNO_SUCCESS();

  if (res == 0) {
    ASSERT_EQ(LIBC_NAMESPACE::pthread_join(tid, nullptr), 0);
    ASSERT_ERRNO_SUCCESS();
    ASSERT_TRUE(args.executed.load());
    ASSERT_EQ(args.policy, SCHED_FIFO);
    ASSERT_EQ(args.priority, 1);
  } else {
    ASSERT_FALSE(args.executed.load());
  }

  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_destroy(&attr), 0);
  ASSERT_ERRNO_SUCCESS();
}

TEST_MAIN() {
  errno = 0;
  run_success_tests();
  run_failure_tests();
  test_sched_inherit();
  test_sched_explicit_success();
  test_sched_realtime_permission();
  return 0;
}
