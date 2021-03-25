// Check that stores in signal handlers are not recorded in origin history.
//
// Origin tracking uses ChainedOriginDepot that is not async signal safe, so we
// do not track origins inside signal handlers.
//
// RUN: %clangxx_dfsan -gmlt -mllvm -dfsan-track-origins=1 -mllvm -dfsan-fast-16-labels=true %s -o %t && \
// RUN:     %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out
//
// RUN: %clangxx_dfsan -gmlt -mllvm -dfsan-instrument-with-call-threshold=0 -mllvm -dfsan-track-origins=1 -mllvm -dfsan-fast-16-labels=true %s -o %t && \
// RUN:     %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out
//
// REQUIRES: x86_64-target-arch

#include <sanitizer/dfsan_interface.h>

#include <signal.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>

int x, y, u;

void SignalHandler(int signo) {
  y = x;
  memcpy(&u, &y, sizeof(int));
}

int main(int argc, char *argv[]) {
  int z = 0;
  dfsan_set_label(8, &z, sizeof(z));
  x = z;

  signal(SIGHUP, SignalHandler);
  kill(getpid(), SIGHUP);
  signal(SIGHUP, SIG_DFL);

  dfsan_print_origin_trace(&u, nullptr);
  return 0;
}

// CHECK: Taint value 0x8 {{.*}} origin tracking ()
// CHECK: Origin value: {{.*}}, Taint value was stored to memory at
// CHECK-NOT: {{.*}} in dfs$SignalHandler {{.*}}origin_with_signals.cpp{{.*}}

// CHECK: #0 {{.*}} in main {{.*}}origin_with_signals.cpp:[[@LINE-14]]

// CHECK: Origin value: {{.*}}, Taint value was created at
// CHECK: #0 {{.*}} in main {{.*}}origin_with_signals.cpp:[[@LINE-18]]
