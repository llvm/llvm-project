//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++11
// UNSUPPORTED: no-threads
// REQUIRES: linux || windows

// __asymmetric_thread_fence_is_asymmetric is always true on Windows. On Linux it starts false, only a heavy
// fence can set it, and after the first heavy fence it tells whether the kernel provides membarrier(2)
// PRIVATE_EXPEDITED -- also when that first heavy fence is issued by many threads at once.

#include <__atomic/asymmetric_thread_fence.h>
#include <atomic>
#include <cassert>
#include <thread>
#include <vector>

#include "make_test_thread.h"

#if defined(__linux__)
#  include <sys/syscall.h>
#  include <unistd.h>
#endif

#if !defined(_WIN32)
static bool kernel_has_private_expedited_membarrier() {
#  if defined(__linux__) && defined(SYS_membarrier)
  long commands = syscall(SYS_membarrier, /* MEMBARRIER_CMD_QUERY */ 0, 0, 0);
  return commands != -1 && (commands & /* MEMBARRIER_CMD_PRIVATE_EXPEDITED */ (1 << 3)) != 0;
#  else
  return false;
#  endif
}
#endif

int main(int, char**) {
#if defined(_WIN32)
  assert(std::__asymmetric_thread_fence_is_asymmetric.load());
  std::__asymmetric_thread_fence_heavy(std::memory_order_seq_cst);
  assert(std::__asymmetric_thread_fence_is_asymmetric.load());
#else
  assert(!std::__asymmetric_thread_fence_is_asymmetric.load());
  std::__asymmetric_thread_fence_light(std::memory_order_seq_cst);
  assert(!std::__asymmetric_thread_fence_is_asymmetric.load());

  // The very first heavy fence comes from 8 threads at once.
  std::atomic<bool> go(false);
  std::vector<std::thread> threads;
  for (int i = 0; i != 8; ++i)
    threads.push_back(support::make_test_thread([&go] {
      while (!go.load())
        std::this_thread::yield();
      std::__asymmetric_thread_fence_heavy(std::memory_order_seq_cst);
      std::__asymmetric_thread_fence_light(std::memory_order_seq_cst);
    }));
  go.store(true);
  for (std::thread& t : threads)
    t.join();

  const bool expected = kernel_has_private_expedited_membarrier();
  assert(std::__asymmetric_thread_fence_is_asymmetric.load() == expected);
  std::__asymmetric_thread_fence_heavy(std::memory_order_seq_cst);
  assert(std::__asymmetric_thread_fence_is_asymmetric.load() == expected);
#endif
  return 0;
}
