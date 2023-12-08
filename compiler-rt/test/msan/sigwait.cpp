// RUN: %clangxx_msan -std=c++11 -O0 -g %s -o %t && %run %t
// RUN: %clangxx_msan -DPOSITIVE -std=c++11 -O0 -g %s -o %t && not %run %t 2>&1 | FileCheck %s

#include <assert.h>
#include <signal.h>
#include <sys/time.h>
#include <sys/wait.h>
#include <unistd.h>

#include <sanitizer/msan_interface.h>

void test_sigwait() {
  sigset_t s;
#ifndef POSITIVE
  sigemptyset(&s);
  sigaddset(&s, SIGUSR1);
#endif
  sigprocmask(SIG_BLOCK, &s, 0);
  // CHECK:  MemorySanitizer: use-of-uninitialized-value

  if (pid_t pid = fork()) {
    kill(pid, SIGUSR1);
    int child_stat;
    wait(&child_stat);
    _exit(!WIFEXITED(child_stat));
  } else {
    int sig;
    int res = sigwait(&s, &sig);
    assert(!res);
    // The following checks that sig is initialized.
    assert(sig == SIGUSR1);
  }
}

int main(void) {
  test_sigwait();
  return 0;
}
