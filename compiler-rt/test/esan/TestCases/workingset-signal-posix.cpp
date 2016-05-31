// RUN: %clang_esan_wset -O0 %s -o %t 2>&1
// RUN: %run %t 2>&1 | FileCheck %s

#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <setjmp.h>
#include <assert.h>

sigjmp_buf mark;

static void SignalHandler(int Sig) {
  if (Sig == SIGSEGV) {
    fprintf(stderr, "Handling SIGSEGV for signal\n");
    siglongjmp(mark, 1);
  }
  exit(1);
}

static void SigactionHandler(int Sig, siginfo_t *Info, void *Ctx) {
  if (Sig == SIGSEGV) {
    fprintf(stderr, "Handling SIGSEGV for sigaction\n");
    siglongjmp(mark, 1);
  }
  exit(1);
}

int main(int argc, char **argv) {
  __sighandler_t Prior = signal(SIGSEGV, SignalHandler);
  assert(Prior == SIG_DFL);
  if (sigsetjmp(mark, 1) == 0)
    *((volatile int *)(ssize_t)argc) = 42; // Raise SIGSEGV
  fprintf(stderr, "Past longjmp for signal\n");

  Prior = signal(SIGSEGV, SIG_DFL);
  assert(Prior == SignalHandler);

  struct sigaction SigAct;
  SigAct.sa_sigaction = SigactionHandler;
  int Res = sigfillset(&SigAct.sa_mask);
  assert(Res == 0);
  SigAct.sa_flags = SA_SIGINFO;
  Res = sigaction(SIGSEGV, &SigAct, NULL);
  assert(Res == 0);

  if (sigsetjmp(mark, 1) == 0)
    *((volatile int *)(ssize_t)argc) = 42; // Raise SIGSEGV
  fprintf(stderr, "Past longjmp for sigaction\n");

  Res = sigaction(SIGSEGV, NULL, &SigAct);
  assert(Res == 0);
  assert(SigAct.sa_sigaction == SigactionHandler);

  return 0;
}
// CHECK:      Handling SIGSEGV for signal
// CHECK-NEXT: Past longjmp for signal
// CHECK-NEXT: Handling SIGSEGV for sigaction
// CHECK-NEXT: Past longjmp for sigaction
// CHECK:      {{.*}} EfficiencySanitizer: the total working set size: {{[0-9][0-9][0-9]}} Bytes ({{[0-9][0-9]}} cache lines)
