// RUN: %clangxx_msan -fsanitize-memory-track-origins=2 -O1 %s -o %t && not %run %t 2>&1 | FileCheck %s

// REQUIRES: x86_64-target-arch

#include <pthread.h>
#include <signal.h>
#include <stdint.h>
#include <ucontext.h>

void handler(int sig, siginfo_t *info, void *uctx) {
  volatile int uninit;
  auto *mctx = &static_cast<ucontext_t *>(uctx)->uc_mcontext;
  auto *fpregs = mctx->fpregs;
  // The member names differ across header versions, but the actual layout
  // is always the same.  So avoid using members, just use arithmetic.
  const uint32_t *after_xmm =
      reinterpret_cast<const uint32_t *>(fpregs + 1) - 24;
  if (after_xmm[12] == FP_XSTATE_MAGIC1)
    reinterpret_cast<_xstate *>(mctx->fpregs)->ymmh.ymmh_space[0] = uninit;
  else
    mctx->gregs[REG_RAX] = uninit;
}

int main(int argc, char **argv) {
  struct sigaction act = {};
  act.sa_sigaction = handler;
  act.sa_flags = SA_SIGINFO;
  sigfillset(&act.sa_mask);
  sigaction(SIGPROF, &act, 0);
  pthread_kill(pthread_self(), SIGPROF);
  return 0;
}

// CHECK: WARNING: MemorySanitizer:
