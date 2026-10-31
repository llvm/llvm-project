//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for clone.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/sched_macros.h"
#include "hdr/signal_macros.h"
#include "hdr/stdint_proxy.h"
#include "hdr/sys_wait_macros.h"
#include "hdr/types/pid_t.h"
#include "hdr/types/size_t.h"
#include "src/__support/OSUtil/syscall.h"
#include "src/sched/clone.h"
#include "src/sys/wait/waitpid.h"
#include "src/unistd/gettid.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

#include <sys/syscall.h>

using namespace LIBC_NAMESPACE::testing::ErrnoSetterMatcher;
using LlvmLibcSchedCloneTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

static constexpr size_t STACK_SIZE = 64 * 1024;
alignas(16) static uint8_t child_stack[STACK_SIZE];

static void *get_child_stack_top() { return child_stack + STACK_SIZE; }

static int simple_child(void *arg) {
  auto *val = reinterpret_cast<uintptr_t *>(arg);
  if (val)
    *val = 42;
  return 123;
}

struct TidArgs {
  pid_t parent_tid = 0;
  pid_t cached_tid = 0;
  pid_t sys_tid = 0;
};

static int check_tid_child(void *arg) {
  auto *args = reinterpret_cast<TidArgs *>(arg);
  pid_t cached = LIBC_NAMESPACE::gettid();
  pid_t sys = LIBC_NAMESPACE::syscall_impl<pid_t>(SYS_gettid);
  if (args) {
    args->cached_tid = cached;
    args->sys_tid = sys;
    if (cached == args->parent_tid)
      return 1;
  }
  if (cached != sys)
    return 2;
  return 0;
}

TEST_F(LlvmLibcSchedCloneTest, BasicProcess) {
  uintptr_t val = 0;
  void *stack_top = get_child_stack_top();
  pid_t pid = LIBC_NAMESPACE::clone(simple_child, stack_top, SIGCHLD, &val);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_GT(pid, 0);

  int status = 0;
  pid_t cpid = LIBC_NAMESPACE::waitpid(pid, &status, 0);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_EQ(cpid, pid);
  ASSERT_TRUE(WIFEXITED(status));
  ASSERT_EQ(WEXITSTATUS(status), 123);
  // Address space was copy-on-write, so parent's val is unchanged.
  ASSERT_EQ(val, static_cast<uintptr_t>(0));
}

TEST_F(LlvmLibcSchedCloneTest, InvalidFlags) {
  void *stack_top = get_child_stack_top();
  // CLONE_THREAD requires CLONE_SIGHAND. Specifying CLONE_THREAD alone fails
  // with EINVAL.
  EXPECT_THAT(
      LIBC_NAMESPACE::clone(simple_child, stack_top, CLONE_THREAD, nullptr),
      Fails(EINVAL));
}

TEST_F(LlvmLibcSchedCloneTest, NullFuncOrStack) {
  void *stack_top = get_child_stack_top();
  EXPECT_THAT(LIBC_NAMESPACE::clone(nullptr, stack_top, SIGCHLD, nullptr),
              Fails(EINVAL));
  EXPECT_THAT(LIBC_NAMESPACE::clone(simple_child, nullptr, SIGCHLD, nullptr),
              Fails(EINVAL));
}

TEST_F(LlvmLibcSchedCloneTest, ChildTidSeparateVm) {
  TidArgs args;
  args.parent_tid = LIBC_NAMESPACE::gettid();
  void *stack_top = get_child_stack_top();
  pid_t pid = LIBC_NAMESPACE::clone(check_tid_child, stack_top, SIGCHLD, &args);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_GT(pid, 0);

  int status = 0;
  pid_t cpid = LIBC_NAMESPACE::waitpid(pid, &status, 0);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_EQ(cpid, pid);
  ASSERT_TRUE(WIFEXITED(status));
  ASSERT_EQ(WEXITSTATUS(status), 0);
  ASSERT_EQ(LIBC_NAMESPACE::gettid(), args.parent_tid);
}

// Emulators struggle with exotic flag combinations (e.g., CLONE_VM or
// CLONE_PARENT_SETTID without CLONE_THREAD).
#ifndef LIBC_TEST_UNDER_EMULATOR
TEST_F(LlvmLibcSchedCloneTest, BasicVmShared) {
  uintptr_t val = 0;
  void *stack_top = get_child_stack_top();
  pid_t pid = LIBC_NAMESPACE::clone(simple_child, stack_top,
                                    CLONE_VM | CLONE_VFORK | SIGCHLD, &val);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_GT(pid, 0);

  int status = 0;
  pid_t cpid = LIBC_NAMESPACE::waitpid(pid, &status, 0);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_EQ(cpid, pid);
  ASSERT_TRUE(WIFEXITED(status));
  ASSERT_EQ(WEXITSTATUS(status), 123);
  // With CLONE_VM, child's modification is visible in the parent.
  ASSERT_EQ(val, static_cast<uintptr_t>(42));
}

TEST_F(LlvmLibcSchedCloneTest, ParentSetTid) {
  void *stack_top = get_child_stack_top();
  pid_t ptid = 0;
  pid_t pid = LIBC_NAMESPACE::clone(
      simple_child, stack_top, CLONE_PARENT_SETTID | SIGCHLD, nullptr, &ptid);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_GT(pid, 0);
  ASSERT_EQ(ptid, pid);

  int status = 0;
  pid_t cpid = LIBC_NAMESPACE::waitpid(pid, &status, 0);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_EQ(cpid, pid);
  ASSERT_TRUE(WIFEXITED(status));
  ASSERT_EQ(WEXITSTATUS(status), 123);
}

TEST_F(LlvmLibcSchedCloneTest, ChildSetTid) {
  void *stack_top = get_child_stack_top();
  pid_t ctid = 0;
  pid_t pid = LIBC_NAMESPACE::clone(
      simple_child, stack_top,
      CLONE_CHILD_SETTID | CLONE_VM | CLONE_VFORK | SIGCHLD, nullptr,
      /*parent_tid=*/nullptr, /*tls=*/nullptr, &ctid);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_GT(pid, 0);
  ASSERT_EQ(ctid, pid);

  int status = 0;
  pid_t cpid = LIBC_NAMESPACE::waitpid(pid, &status, 0);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_EQ(cpid, pid);
  ASSERT_TRUE(WIFEXITED(status));
  ASSERT_EQ(WEXITSTATUS(status), 123);
}

TEST_F(LlvmLibcSchedCloneTest, ChildTidSharedVm) {
  TidArgs args;
  args.parent_tid = LIBC_NAMESPACE::gettid();
  void *stack_top = get_child_stack_top();
  pid_t pid = LIBC_NAMESPACE::clone(check_tid_child, stack_top,
                                    CLONE_VM | CLONE_VFORK | SIGCHLD, &args);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_GT(pid, 0);

  int status = 0;
  pid_t cpid = LIBC_NAMESPACE::waitpid(pid, &status, 0);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_EQ(cpid, pid);
  ASSERT_TRUE(WIFEXITED(status));
  ASSERT_EQ(WEXITSTATUS(status), 0);
  ASSERT_EQ(args.cached_tid, args.sys_tid);
  ASSERT_NE(args.cached_tid, args.parent_tid);
  ASSERT_EQ(LIBC_NAMESPACE::gettid(), args.parent_tid);
}
#endif // LIBC_TEST_UNDER_EMULATOR
