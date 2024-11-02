// RUN: %clangxx_msan -fno-sanitize-memory-param-retval -std=c++11 -O0 %s -o %t && %run %t
//
// Test that va_arg shadow from a signal handler does not leak outside.

// Reported deadly signal due to stack-overflow
// XFAIL: netbsd

#include <signal.h>
#include <stdarg.h>
#include <sanitizer/msan_interface.h>
#include <assert.h>
#include <sys/time.h>
#include <stdio.h>

const int kSigCnt = 200;

void f(bool poisoned, int n, ...) {
  va_list vl;
  va_start(vl, n);
  for (int i = 0; i < n; ++i) {
    void *p = va_arg(vl, void *);
    if (!poisoned)
      assert(__msan_test_shadow(&p, sizeof(p)) == -1);
  }
  va_end(vl);
}

int sigcnt;

void SignalHandler(int signo) {
  assert(signo == SIGPROF);
  void *p;
  void **volatile q = &p;
  f(true, 10,
    *q, *q, *q, *q, *q,
    *q, *q, *q, *q, *q);
  ++sigcnt;
}

int main() {
  signal(SIGPROF, SignalHandler);

  itimerval itv;
  itv.it_interval.tv_sec = 0;
  itv.it_interval.tv_usec = 100;
  itv.it_value.tv_sec = 0;
  itv.it_value.tv_usec = 100;
  setitimer(ITIMER_PROF, &itv, NULL);

  void *p;
  void **volatile q = &p;

  do {
    f(false, 20,
      nullptr, nullptr, nullptr, nullptr, nullptr,
      nullptr, nullptr, nullptr, nullptr, nullptr,
      nullptr, nullptr, nullptr, nullptr, nullptr,
      nullptr, nullptr, nullptr, nullptr, nullptr);
    f(true, 20,
      *q, *q, *q, *q, *q,
      *q, *q, *q, *q, *q,
      *q, *q, *q, *q, *q,
      *q, *q, *q, *q, *q);
  } while (sigcnt < kSigCnt);

  itv.it_interval.tv_sec = 0;
  itv.it_interval.tv_usec = 0;
  itv.it_value.tv_sec = 0;
  itv.it_value.tv_usec = 0;
  setitimer(ITIMER_PROF, &itv, NULL);

  signal(SIGPROF, SIG_DFL);
  return 0;
}
