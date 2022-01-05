//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: libcpp-has-no-threads

// This test uses the POSIX header <sys/time.h> which Windows doesn't provide
// UNSUPPORTED: windows

// Until 58a0a70fb2f1, this_thread::sleep_for misbehaves when interrupted by
// a signal, as tested here. Disable the test on the corresponding system
// libraries.
// XFAIL: use_system_cxx_lib && target={{.+}}-apple-macosx10.{{9|10|11}}

// ALLOW_RETRIES: 3

// <thread>

// template <class Rep, class Period>
//   void sleep_for(const chrono::duration<Rep, Period>& rel_time);

// This test ensures that we sleep for the right amount of time even when
// we get interrupted by a signal, as fixed in 58a0a70fb2f1.

#include <thread>
#include <cassert>
#include <chrono>
#include <cstring> // for std::memset

#include <signal.h>
#include <sys/time.h>

#include "test_macros.h"

void sig_action(int) {}

int main(int, char**)
{
  int ec;
  struct sigaction action;
  action.sa_handler = &sig_action;
  sigemptyset(&action.sa_mask);
  action.sa_flags = 0;

  ec = sigaction(SIGALRM, &action, nullptr);
  assert(!ec);

  struct itimerval it;
  std::memset(&it, 0, sizeof(itimerval));
  it.it_value.tv_sec = 0;
  it.it_value.tv_usec = 250000;
  // This will result in a SIGALRM getting fired resulting in the nanosleep
  // inside sleep_for getting EINTR.
  ec = setitimer(ITIMER_REAL, &it, nullptr);
  assert(!ec);

  typedef std::chrono::system_clock Clock;
  typedef Clock::time_point time_point;
  std::chrono::milliseconds ms(500);
  time_point t0 = Clock::now();
  std::this_thread::sleep_for(ms);
  time_point t1 = Clock::now();
  // NOTE: Operating systems are (by default) best effort and therefore we may
  // have slept longer, perhaps much longer than we requested.
  assert(t1 - t0 >= ms);

  return 0;
}
