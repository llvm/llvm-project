// RUN: %clangxx_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s

// Always enable asserts.
#ifdef NDEBUG
#undef NDEBUG
#endif

#include <assert.h>
#include <stdlib.h>
#include <signal.h>
#include <unistd.h>
#include <stdio.h>

static bool signal_handler_ran = false;

void do_nothing_signal_handler(int signum) {
  write(1, "HANDLER\n", 8);
  signal_handler_ran = true;
}

int main() {
  const int kSignalToTest = SIGSYS;
  assert(SIG_ERR != signal(kSignalToTest, do_nothing_signal_handler));
  sigset_t empty_set;
  assert(0 == sigemptyset(&empty_set));
  sigset_t one_signal = empty_set;
  assert(0 == sigaddset(&one_signal, kSignalToTest));
  sigset_t old_set;
  assert(0 == sigprocmask(SIG_BLOCK, &one_signal, &old_set));
  raise(kSignalToTest);
  assert(!signal_handler_ran);
  sigset_t all_but_one;
  assert(0 == sigfillset(&all_but_one));
  assert(0 == sigdelset(&all_but_one, kSignalToTest));
  sigsuspend(&all_but_one);
  assert(signal_handler_ran);

  // Restore the original set.
  assert(0 == sigprocmask(SIG_SETMASK, &old_set, NULL));
  printf("DONE\n");
}

// CHECK: HANDLER
// CHECK: DONE
