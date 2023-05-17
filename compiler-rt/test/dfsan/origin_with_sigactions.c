// Check that stores in signal handlers are not recorded in origin history.
//
// Origin tracking uses ChainedOriginDepot that is not async signal safe, so we
// do not track origins inside signal handlers.
//
// RUN: %clang_dfsan -gmlt -DUSE_SIGNAL_ACTION -mllvm -dfsan-track-origins=1 %s -o %t && \
// RUN:      %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out
//
// RUN: %clang_dfsan -gmlt -DUSE_SIGNAL_ACTION -mllvm -dfsan-instrument-with-call-threshold=0 -mllvm -dfsan-track-origins=1 %s -o %t && \
// RUN:     %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out
//
// RUN: %clang_dfsan -gmlt -mllvm -dfsan-track-origins=1 %s -o %t && \
// RUN:     %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out
//
// RUN: %clang_dfsan -gmlt -mllvm -dfsan-instrument-with-call-threshold=0 -mllvm -dfsan-track-origins=1 %s -o %t && \
// RUN:     %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out

#include <sanitizer/dfsan_interface.h>

#include <assert.h>
#include <signal.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>

int x, y, u;

void CopyXtoYtoU() {
  y = x;
  memcpy(&u, &y, sizeof(int));
}

void SignalHandler(int signo) {
  CopyXtoYtoU();
}

void SignalAction(int signo, siginfo_t *si, void *uc) {
  CopyXtoYtoU();
}

int main(int argc, char *argv[]) {
  int z = 1;
  dfsan_set_label(8, &z, sizeof(z));
  x = z;

  struct sigaction psa = {};
#ifdef USE_SIGNAL_ACTION
  psa.sa_flags = SA_SIGINFO;
  psa.sa_sigaction = SignalAction;
#else
  psa.sa_flags = 0;
  psa.sa_handler = SignalHandler;
#endif
  sigaction(SIGHUP, &psa, NULL);
  kill(getpid(), SIGHUP);
  signal(SIGHUP, SIG_DFL);

  assert(x == 1);
  assert(y == 1);
  assert(u == 1);

  dfsan_print_origin_trace(&u, NULL);
  return 0;
}

// CHECK: Taint value 0x8 {{.*}} origin tracking ()
// CHECK: Origin value: {{.*}}, Taint value was stored to memory at
// CHECK-NOT: {{.*}} in CopyXtoYtoU.dfsan {{.*}}origin_with_sigactions.c{{.*}}

// CHECK: #0 {{.*}} in main {{.*}}origin_with_sigactions.c:[[@LINE-26]]

// CHECK: Origin value: {{.*}}, Taint value was created at
// CHECK: #0 {{.*}} in main {{.*}}origin_with_sigactions.c:[[@LINE-30]]
