// RUN: %clang_dfsan -DUSE_SIGNAL_ACTION -Wno-error=int-conversion %s -o %t && %run %t
// RUN: %clang_dfsan -Wno-error=int-conversion %s -o %t && %run %t

#include <sanitizer/dfsan_interface.h>

#include <assert.h>
#include <signal.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>

volatile int x;
volatile int z = 1;

void SignalHandler(int signo) {
  assert(dfsan_get_label(signo) == 0);
  x = z;
}

void SignalAction(int signo, siginfo_t *si, void *uc) {
  assert(dfsan_get_label(signo) == 0);
  assert(dfsan_get_label(si) == 0);
  assert(dfsan_get_label(uc) == 0);
  assert(0 == dfsan_read_label(si, sizeof(*si)));
  assert(0 == dfsan_read_label(uc, sizeof(ucontext_t)));
  x = z;
}

int main(int argc, char *argv[]) {
  dfsan_set_label(8, (void *)&z, sizeof(z));

  struct sigaction sa = {};
#ifdef USE_SIGNAL_ACTION
  sa.sa_flags = SA_SIGINFO;
  sa.sa_sigaction = SignalAction;
#else
  sa.sa_handler = SignalHandler;
#endif
  int r = sigaction(SIGHUP, &sa, NULL);
  assert(dfsan_get_label(r) == 0);

  kill(getpid(), SIGHUP);
  signal(SIGHUP, SIG_DFL);

  assert(x == 1);

  return 0;
}
