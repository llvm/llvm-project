//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <signal.h>
#include <stdio.h>
#include <vector>

static int signal_counter = 0;

static void count_signal(int signo) {
  ++signal_counter;
  printf("Signal %d\n", signo);
}

static void raise_signals() {
  std::vector<int> signals(
      {SIGSEGV, SIGUSR1, SIGUSR2, SIGALRM, SIGFPE, SIGBUS, SIGINT, SIGHUP});

  for (int signal_num : signals) {
    signal(signal_num, count_signal);
  }

  for (int signal_num : signals) {
    raise(signal_num);
  }
}

int main() {
  raise_signals();
  return signal_counter;
}
