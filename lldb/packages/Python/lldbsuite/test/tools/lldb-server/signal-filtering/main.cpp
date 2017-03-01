//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
