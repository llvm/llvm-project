//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++11
// UNSUPPORTED: no-threads
// UNSUPPORTED: windows

// After the parent has issued a heavy fence (which enables the asymmetric path where the system supports
// it), a forked child can still use both fences, and they still order memory.

#include <__atomic/asymmetric_thread_fence.h>
#include <atomic>
#include <cassert>
#include <sys/wait.h>
#include <unistd.h>

#include "store_buffering.h"

int main(int, char**) {
  std::__asymmetric_thread_fence_heavy(std::memory_order_seq_cst);

  pid_t pid = fork();
  assert(pid != -1);
  if (pid == 0) {
    std::__asymmetric_thread_fence_light(std::memory_order_seq_cst);
    std::__asymmetric_thread_fence_heavy(std::memory_order_seq_cst);
    check_store_buffering(light, heavy, std::memory_order_seq_cst, 20000);
    _exit(0);
  }

  int status = 0;
  assert(waitpid(pid, &status, 0) == pid);
  assert(WIFEXITED(status) && WEXITSTATUS(status) == 0);
  return 0;
}
