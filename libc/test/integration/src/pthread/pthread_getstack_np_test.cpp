//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Integration tests for pthread_getstack_np.
///
//===----------------------------------------------------------------------===//

#include "hdr/pthread_macros.h"
#include "hdr/stdint_proxy.h"
#include "hdr/sys_mman_macros.h"
#include "src/pthread/pthread_attr_destroy.h"
#include "src/pthread/pthread_attr_init.h"
#include "src/pthread/pthread_attr_setstack.h"
#include "src/pthread/pthread_create.h"
#include "src/pthread/pthread_getstack_np.h"
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

static void *child_func(void *) {
  void *stackaddr = nullptr;
  size_t stacksize = 0;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_getstack_np(LIBC_NAMESPACE::pthread_self(),
                                                &stackaddr, &stacksize),
            0);
  ASSERT_NE(stackaddr, static_cast<void *>(nullptr));
  ASSERT_NE(stacksize, static_cast<size_t>(PTHREAD_STACK_DYNAMIC_NP));
  ASSERT_TRUE(stacksize > 0);

  uintptr_t local_var_addr = reinterpret_cast<uintptr_t>(&stackaddr);
  uintptr_t stack_low = reinterpret_cast<uintptr_t>(stackaddr);
  uintptr_t stack_high = stack_low + stacksize;
  ASSERT_TRUE(local_var_addr >= stack_low);
  ASSERT_TRUE(local_var_addr < stack_high);

  check_readable(stackaddr, stacksize);

  return nullptr;
}

static void test_main_thread() {
  void *stackaddr = nullptr;
  size_t stacksize = 1234;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_getstack_np(LIBC_NAMESPACE::pthread_self(),
                                                &stackaddr, &stacksize),
            0);
  ASSERT_NE(stackaddr, static_cast<void *>(nullptr));
  ASSERT_EQ(stacksize, static_cast<size_t>(PTHREAD_STACK_DYNAMIC_NP));
  ASSERT_EQ(reinterpret_cast<uintptr_t>(stackaddr) %
                LIBC_NAMESPACE::sysconf(_SC_PAGESIZE),
            static_cast<uintptr_t>(0));

  // Stack grows downwards on linux, so local variables on the main thread stack
  // should be at addresses lower than the initial stack top address.
  uintptr_t local_var_addr = reinterpret_cast<uintptr_t>(&stackaddr);
  ASSERT_TRUE(local_var_addr < reinterpret_cast<uintptr_t>(stackaddr));

  // The stack region between the current stack frame and the top of the stack
  // should be readable.
  check_readable(&stackaddr,
                 reinterpret_cast<uintptr_t>(stackaddr) - local_var_addr);
}

static void test_child_thread_default() {
  pthread_t th;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_create(&th, nullptr, child_func, nullptr),
            0);
  void *retval = nullptr;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_join(th, &retval), 0);
}

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

  pthread_t th;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_create(&th, &attr, child_func, nullptr), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_destroy(&attr), 0);

  void *stackaddr = nullptr;
  size_t stacksize = 0;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_getstack_np(th, &stackaddr, &stacksize), 0);
  ASSERT_EQ(stackaddr, custom_stack);
  ASSERT_EQ(stacksize, custom_stacksize);
  check_readable(stackaddr, stacksize);

  void *retval = nullptr;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_join(th, &retval), 0);
  ASSERT_EQ(LIBC_NAMESPACE::munmap(custom_stack, custom_stacksize), 0);
}

TEST_MAIN() {
  test_main_thread();
  test_child_thread_default();
  test_child_thread_custom_stack();
  return 0;
}
