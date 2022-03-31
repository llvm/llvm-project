// RUN: %clang_dfsan -fno-sanitize=dataflow -O2 -fPIE -DCALLBACKS -c %s -o %t-callbacks.o
// RUN: %clang_dfsan -fsanitize-ignorelist=%S/Inputs/flags_abilist.txt -O2 -mllvm -dfsan-conditional-callbacks %s %t-callbacks.o -o %t
// RUN: %run %t FooBarBaz 2>&1 | FileCheck %s
//
// REQUIRES: x86_64-target-arch

#include <assert.h>
#include <sanitizer/dfsan_interface.h>
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>

#ifdef CALLBACKS
// Compile this code without DFSan to avoid recursive instrumentation.

void my_dfsan_conditional_callback(dfsan_label Label, dfsan_origin Origin) {
  assert(Label != 0);
  assert(Origin == 0);

  static int Count = 0;
  switch (Count++) {
  case 0:
    assert(Label == 1);
    break;
  case 1:
    assert(Label == 4);
    break;
  default:
    break;
  }

  fprintf(stderr, "Label %u used as condition\n", Label);
}

#else
// Compile this code with DFSan and -dfsan-conditional-callbacks to insert the
// callbacks.

extern void my_dfsan_conditional_callback(dfsan_label Label,
                                          dfsan_origin Origin);

volatile int x = 0;
volatile int y = 1;
volatile int z = 0;

void SignalHandler(int signo) {
  assert(dfsan_get_label(x) == 0);
  assert(dfsan_get_label(y) != 0);
  assert(dfsan_get_label(z) != 0);
  // Running the conditional callback from a signal handler is risky,
  // because the code must be written with signal handler context in mind.
  // Instead dfsan_get_labels_in_signal_conditional() will indicate labels
  // used in conditions inside signal handlers.
  // CHECK-NOT: Label 8 used as condition
  if (z != 0) {
    x = y;
  }
}

int main(int Argc, char *Argv[]) {
  assert(Argc >= 1);
  int unknown = (Argv[0][0] != 0) ? 1 : 0;
  dfsan_set_label(1, &unknown, sizeof(unknown));

  dfsan_set_conditional_callback(my_dfsan_conditional_callback);

  // CHECK: Label 1 used as condition
  if (unknown) {
    z = 42;
  }

  assert(dfsan_get_labels_in_signal_conditional() == 0);
  dfsan_set_label(4, (void *)&y, sizeof(y));
  dfsan_set_label(8, (void *)&z, sizeof(z));

  struct sigaction sa = {};
  sa.sa_handler = SignalHandler;
  int r = sigaction(SIGHUP, &sa, NULL);
  assert(dfsan_get_label(r) == 0);

  kill(getpid(), SIGHUP);
  signal(SIGHUP, SIG_DFL);

  assert(dfsan_get_labels_in_signal_conditional() == 8);
  assert(x == 1);
  // CHECK: Label 4 used as condition
  if (x != 0) {
    z = 123;
  }
  // Flush should clear the conditional signals seen.
  dfsan_flush();
  assert(dfsan_get_labels_in_signal_conditional() == 0);
  return 0;
}

#endif // #ifdef CALLBACKS
