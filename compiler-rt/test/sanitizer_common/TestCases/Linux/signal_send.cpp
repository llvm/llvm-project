// RUN: %clangxx -std=c++11 -O0 -g %s -o %t && %run %t 2>&1 | FileCheck %s

// sigandset is glibc specific.
// UNSUPPORTED: android, freebsd, netbsd

#include <assert.h>
#include <signal.h>
#include <stdio.h>
#include <sys/time.h>
#include <sys/wait.h>
#include <unistd.h>

sigset_t mkset(int n, ...) {
  sigset_t s;
  int res = 0;
  res |= sigemptyset(&s);
  va_list va;
  va_start(va, n);
  while (n--) {
    res |= sigaddset(&s, va_arg(va, int));
  }
  va_end(va);
  assert(!res);
  return s;
}

sigset_t sigset_or(sigset_t first, sigset_t second) {
  sigset_t out;
  int res = sigorset(&out, &first, &second);
  assert(!res);
  return out;
}

sigset_t sigset_and(sigset_t first, sigset_t second) {
  sigset_t out;
  int res = sigandset(&out, &first, &second);
  assert(!res);
  return out;
}

int fork_and_signal(sigset_t s) {
  if (pid_t pid = fork()) {
    kill(pid, SIGUSR1);
    kill(pid, SIGUSR2);
    int child_stat;
    wait(&child_stat);
    return !WIFEXITED(child_stat);
  } else {
    int sig;
    int res = sigwait(&s, &sig);
    assert(!res);
    fprintf(stderr, "died with sig %d\n", sig);
    _exit(0);
  }
}

void test_sigwait() {
  // test sigorset... s should now contain SIGUSR1 | SIGUSR2
  sigset_t s = sigset_or(mkset(1, SIGUSR1), mkset(1, SIGUSR2));
  sigprocmask(SIG_BLOCK, &s, 0);
  int res;
  res = fork_and_signal(s);
  fprintf(stderr, "fork_and_signal with SIGUSR1,2: %d\n", res);
  // CHECK: died with sig 10
  // CHECK: fork_and_signal with SIGUSR1,2: 0

  // test sigandset... s should only have SIGUSR2 now
  s = sigset_and(s, mkset(1, SIGUSR2));
  res = fork_and_signal(s);
  fprintf(stderr, "fork_and_signal with SIGUSR2: %d\n", res);
  // CHECK: died with sig 12
  // CHECK: fork_and_signal with SIGUSR2: 0
}

int main(void) {
  test_sigwait();
  return 0;
}
