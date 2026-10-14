//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Test handling of overly aligned thread local data.
///
//===----------------------------------------------------------------------===//

#include "hdr/stdint_proxy.h"
#include "src/pthread/pthread_create.h"
#include "src/pthread/pthread_join.h"
#include "test/IntegrationTest/test.h"

#include <pthread.h>

constexpr size_t ALIGN1 = 64;
constexpr size_t ALIGN2 = 128;
constexpr size_t ALIGN3 = 256;

alignas(ALIGN1) static thread_local int aligned_var1 = 123;
alignas(ALIGN2) static thread_local int aligned_var2 = 456;
alignas(ALIGN3) static thread_local int aligned_bss_var;

struct ThreadTlsPointers {
  void *ptr1 = nullptr;
  void *ptr2 = nullptr;
  void *ptr3 = nullptr;
};

static void *thread_func(void *arg) {
  auto *ptrs = static_cast<ThreadTlsPointers *>(arg);

  ASSERT_EQ(reinterpret_cast<uintptr_t>(&aligned_var1) % ALIGN1,
            static_cast<uintptr_t>(0));
  ASSERT_EQ(reinterpret_cast<uintptr_t>(&aligned_var2) % ALIGN2,
            static_cast<uintptr_t>(0));
  ASSERT_EQ(reinterpret_cast<uintptr_t>(&aligned_bss_var) % ALIGN3,
            static_cast<uintptr_t>(0));
  ASSERT_EQ(aligned_var1, 123);
  ASSERT_EQ(aligned_var2, 456);
  ASSERT_EQ(aligned_bss_var, 0);

  ptrs->ptr1 = &aligned_var1;
  ptrs->ptr2 = &aligned_var2;
  ptrs->ptr3 = &aligned_bss_var;

  aligned_var1 = 789;
  aligned_var2 = 101112;
  aligned_bss_var = 131415;

  ASSERT_EQ(aligned_var1, 789);
  ASSERT_EQ(aligned_var2, 101112);
  ASSERT_EQ(aligned_bss_var, 131415);

  return nullptr;
}

TEST_MAIN() {
  ASSERT_EQ(reinterpret_cast<uintptr_t>(&aligned_var1) % ALIGN1,
            static_cast<uintptr_t>(0));
  ASSERT_EQ(reinterpret_cast<uintptr_t>(&aligned_var2) % ALIGN2,
            static_cast<uintptr_t>(0));
  ASSERT_EQ(reinterpret_cast<uintptr_t>(&aligned_bss_var) % ALIGN3,
            static_cast<uintptr_t>(0));
  ASSERT_EQ(aligned_var1, 123);
  ASSERT_EQ(aligned_var2, 456);
  ASSERT_EQ(aligned_bss_var, 0);

  pthread_t th1;
  pthread_t th2;
  ThreadTlsPointers th1_ptrs;
  ThreadTlsPointers th2_ptrs;
  void *retval = nullptr;

  ASSERT_EQ(
      LIBC_NAMESPACE::pthread_create(&th1, nullptr, thread_func, &th1_ptrs), 0);
  ASSERT_EQ(
      LIBC_NAMESPACE::pthread_create(&th2, nullptr, thread_func, &th2_ptrs), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_join(th1, &retval), 0);
  ASSERT_EQ(retval, nullptr);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_join(th2, &retval), 0);
  ASSERT_EQ(retval, nullptr);

  // Child thread pointers must not match main thread.
  ASSERT_NE(th1_ptrs.ptr1, static_cast<void *>(&aligned_var1));
  ASSERT_NE(th1_ptrs.ptr2, static_cast<void *>(&aligned_var2));
  ASSERT_NE(th1_ptrs.ptr3, static_cast<void *>(&aligned_bss_var));

  ASSERT_NE(th2_ptrs.ptr1, static_cast<void *>(&aligned_var1));
  ASSERT_NE(th2_ptrs.ptr2, static_cast<void *>(&aligned_var2));
  ASSERT_NE(th2_ptrs.ptr3, static_cast<void *>(&aligned_bss_var));

  // Child thread pointers must not match each other.
  ASSERT_NE(th1_ptrs.ptr1, th2_ptrs.ptr1);
  ASSERT_NE(th1_ptrs.ptr2, th2_ptrs.ptr2);
  ASSERT_NE(th1_ptrs.ptr3, th2_ptrs.ptr3);

  // Child thread modifications must not affect main thread.
  ASSERT_EQ(aligned_var1, 123);
  ASSERT_EQ(aligned_var2, 456);
  ASSERT_EQ(aligned_bss_var, 0);

  return 0;
}
