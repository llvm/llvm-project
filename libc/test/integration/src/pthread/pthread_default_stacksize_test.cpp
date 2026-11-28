//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Integration test for default thread stack size derived from startup
/// RLIMIT_STACK.
///
//===----------------------------------------------------------------------===//

#include "hdr/sys_resource_macros.h"
#include "hdr/sys_wait_macros.h"
#include "hdr/types/pid_t.h"
#include "hdr/types/pthread_attr_t.h"
#include "hdr/types/pthread_t.h"
#include "hdr/types/size_t.h"
#include "hdr/types/struct_rlimit.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/threads/stack_size.h"
#include "src/pthread/pthread_attr_destroy.h"
#include "src/pthread/pthread_attr_getstacksize.h"
#include "src/pthread/pthread_attr_init.h"
#include "src/pthread/pthread_create.h"
#include "src/pthread/pthread_getattr_np.h"
#include "src/pthread/pthread_getstack_np.h"
#include "src/pthread/pthread_join.h"
#include "src/pthread/pthread_self.h"
#include "src/stdlib/_Exit.h"
#include "src/sys/resource/getrlimit.h"
#include "src/sys/resource/setrlimit.h"
#include "src/sys/wait/waitpid.h"
#include "src/threads/thrd_create.h"
#include "src/threads/thrd_join.h"
#include "src/unistd/execve.h"
#include "src/unistd/fork.h"
#include "test/IntegrationTest/test.h"

#include <linux/param.h> // For EXEC_PAGESIZE.
#include <threads.h>     // For thrd_t and thrd_success.

static void *pthread_probe(void *arg) {
  auto *out_size = static_cast<size_t *>(arg);
  pthread_attr_t attr;
  ASSERT_EQ(
      LIBC_NAMESPACE::pthread_getattr_np(LIBC_NAMESPACE::pthread_self(), &attr),
      0);
  size_t getattr_size = 0;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_getstacksize(&attr, &getattr_size), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_destroy(&attr), 0);

  void *stackaddr = nullptr;
  size_t getstack_size = 0;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_getstack_np(LIBC_NAMESPACE::pthread_self(),
                                                &stackaddr, &getstack_size),
            0);
  ASSERT_EQ(getattr_size, getstack_size);
  *out_size = getattr_size;
  return nullptr;
}

static int c11_thread_probe(void *arg) {
  auto *out_size = static_cast<size_t *>(arg);
  void *stackaddr = nullptr;
  size_t stacksize = 0;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_getstack_np(LIBC_NAMESPACE::pthread_self(),
                                                &stackaddr, &stacksize),
            0);
  *out_size = stacksize;
  return 0;
}

static void verify_default_stacksize(size_t expected) {
  pthread_attr_t attr;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_init(&attr), 0);
  size_t attr_size = 0;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_getstacksize(&attr, &attr_size), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_destroy(&attr), 0);
  ASSERT_EQ(attr_size, expected);

  size_t pt_size = 0;
  pthread_t pt;
  ASSERT_EQ(
      LIBC_NAMESPACE::pthread_create(&pt, nullptr, pthread_probe, &pt_size), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_join(pt, nullptr), 0);
  ASSERT_EQ(pt_size, expected);

  size_t ct_size = 0;
  thrd_t ct;
  ASSERT_EQ(LIBC_NAMESPACE::thrd_create(&ct, c11_thread_probe, &ct_size),
            static_cast<int>(thrd_success));
  ASSERT_EQ(LIBC_NAMESPACE::thrd_join(ct, nullptr),
            static_cast<int>(thrd_success));
  ASSERT_EQ(ct_size, expected);
}

constexpr size_t FINITE_SIZE = 512 * 1024;

static bool global_ctor_checked = false;

__attribute__((constructor)) static void
check_global_ctor(int argc, char **argv, char **) {
  if (argc != 2 || LIBC_NAMESPACE::cpp::string_view(argv[1]) != "global_ctor")
    return;

  verify_default_stacksize(FINITE_SIZE);
  global_ctor_checked = true;
}

static void spawn_child_with_rlimit(const char *self_path, char **envp,
                                    rlim_t rlim_cur, const char *mode) {
  pid_t pid = LIBC_NAMESPACE::fork();
  ASSERT_TRUE(pid >= 0);
  if (pid == 0) {
    struct rlimit rl;
    if (LIBC_NAMESPACE::getrlimit(RLIMIT_STACK, &rl) != 0)
      LIBC_NAMESPACE::_Exit(1);
    rl.rlim_cur = rlim_cur;
    if (LIBC_NAMESPACE::setrlimit(RLIMIT_STACK, &rl) != 0)
      LIBC_NAMESPACE::_Exit(2);

    char *const child_argv[] = {
        const_cast<char *>(self_path),
        const_cast<char *>(mode),
        nullptr,
    };
    LIBC_NAMESPACE::execve(self_path, child_argv, envp);
    LIBC_NAMESPACE::_Exit(3);
  }

  int status = 0;
  ASSERT_EQ(LIBC_NAMESPACE::waitpid(pid, &status, 0), pid);
  ASSERT_TRUE(WIFEXITED(status));
  ASSERT_EQ(WEXITSTATUS(status), 0);
}

TEST_MAIN(int argc, char **argv, char **envp) {
  constexpr size_t RUNTIME_NEW_SIZE = 256 * 1024;
  constexpr size_t UNALIGNED_SIZE = 256 * 1024 + 123;
  constexpr size_t PAGE_MASK = static_cast<size_t>(EXEC_PAGESIZE) - 1;

  if (argc == 2) {
    LIBC_NAMESPACE::cpp::string_view mode(argv[1]);
    if (mode == "unlimited") {
      verify_default_stacksize(LIBC_NAMESPACE::detail::FALLBACK_STACKSIZE);
      return 0;
    }
    if (mode == "finite_and_runtime_change") {
      // Lower RLIMIT_STACK before making any thread or attribute calls in main;
      // the default stack size must still reflect the startup RLIMIT_STACK.
      struct rlimit rl;
      ASSERT_EQ(LIBC_NAMESPACE::getrlimit(RLIMIT_STACK, &rl), 0);
      ASSERT_EQ(rl.rlim_cur, static_cast<rlim_t>(FINITE_SIZE));
      rl.rlim_cur = RUNTIME_NEW_SIZE;
      ASSERT_EQ(LIBC_NAMESPACE::setrlimit(RLIMIT_STACK, &rl), 0);

      verify_default_stacksize(FINITE_SIZE);
      return 0;
    }
    if (mode == "global_ctor") {
      ASSERT_TRUE(global_ctor_checked);
      return 0;
    }
    if (mode == "unaligned") {
      verify_default_stacksize((UNALIGNED_SIZE + PAGE_MASK) & ~PAGE_MASK);
      return 0;
    }
    return 1;
  }

  struct rlimit rl;
  ASSERT_EQ(LIBC_NAMESPACE::getrlimit(RLIMIT_STACK, &rl), 0);
  verify_default_stacksize(
      LIBC_NAMESPACE::detail::stacksize_from_rlimit(rl.rlim_cur));

  if (rl.rlim_max == RLIM_INFINITY)
    spawn_child_with_rlimit(argv[0], envp, RLIM_INFINITY, "unlimited");

  if (rl.rlim_max >= FINITE_SIZE) {
    spawn_child_with_rlimit(argv[0], envp, FINITE_SIZE,
                            "finite_and_runtime_change");
    spawn_child_with_rlimit(argv[0], envp, FINITE_SIZE, "global_ctor");
  }

  if (rl.rlim_max >= UNALIGNED_SIZE)
    spawn_child_with_rlimit(argv[0], envp, UNALIGNED_SIZE, "unaligned");

  return 0;
}
