// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: linux && target={{aarch64-.+}}

// pthread_cancel in case of glibc calls _Unwind_ForcedUnwind from a signal on
// the child_thread. This test ensures sigretrun is handled correctly (see:
// UnwindCursor<A, R>::setInfoForSigReturn).

#include <cstdlib> // defines __BIONIC__

// Android/Bionic does not support pthread_cancel.
#ifdef __BIONIC__
int main() {
  return 0;
}
#else

#include <chrono>
#include <condition_variable>
#include <pthread.h>
#include <unistd.h>

using namespace std::chrono_literals;

std::condition_variable cv;
std::mutex cv_m;
bool thread_ready = false;

static void* test(void* arg) {
  (void)arg;
  thread_ready = true;
  cv.notify_all();

  // This must be a pthread cancellation point.
  while (1)
    sleep(100);

  return (void*)1;
}

int main() {
  pthread_t child_thread;
  std::unique_lock<std::mutex> lk(cv_m);
  pthread_create(&child_thread, 0, test, (void*)0);

  if (!cv.wait_for(lk, 100ms, [] { return thread_ready; }))
    return -1;

  pthread_cancel(child_thread);
  pthread_join(child_thread, NULL);
  return 0;
}
#endif
