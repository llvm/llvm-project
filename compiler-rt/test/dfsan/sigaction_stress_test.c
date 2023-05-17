// RUN: %clangxx_dfsan %s -o %t && %run %t
// RUN: %clangxx_dfsan -mllvm -dfsan-track-origins=1 %s -o %t && %run %t
// RUN: %clangxx_dfsan -mllvm -dfsan-track-origins=1 -mllvm -dfsan-instrument-with-call-threshold=0 %s -o %t && %run %t
//
// Test that the state of shadows from a sigaction handler are consistent.

#include <signal.h>
#include <stdarg.h>
#include <sanitizer/dfsan_interface.h>
#include <assert.h>
#include <sys/time.h>
#include <stdio.h>

const int kSigCnt = 200;
int x = 0;

__attribute__((noinline))
int f(int a) {
  return a;
}

__attribute__((noinline))
void g() {
  int r = f(x);
  const dfsan_label r_label = dfsan_get_label(r);
  assert(r_label == 8 || r_label == 0);
  return;
}

int sigcnt;

void SignalHandler(int signo) {
  assert(signo == SIGPROF);
  int a = 0;
  dfsan_set_label(4, &a, sizeof(a));
  (void)f(a);
  ++sigcnt;
}

int main() {
  struct sigaction psa = {};
  psa.sa_handler = SignalHandler;
  int r = sigaction(SIGPROF, &psa, NULL);

  itimerval itv;
  itv.it_interval.tv_sec = 0;
  itv.it_interval.tv_usec = 100;
  itv.it_value.tv_sec = 0;
  itv.it_value.tv_usec = 100;
  setitimer(ITIMER_PROF, &itv, NULL);

  dfsan_set_label(8, &x, sizeof(x));
  do {
    g();
  } while (sigcnt < kSigCnt);

  itv.it_interval.tv_sec = 0;
  itv.it_interval.tv_usec = 0;
  itv.it_value.tv_sec = 0;
  itv.it_value.tv_usec = 0;
  setitimer(ITIMER_PROF, &itv, NULL);

  signal(SIGPROF, SIG_DFL);
  return 0;
}
