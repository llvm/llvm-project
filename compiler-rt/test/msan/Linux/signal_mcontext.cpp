// RUN: %clangxx_msan -fsanitize-memory-track-origins=2 -O1 %s -o %t && %run %t 2>&1 | FileCheck %s

#include <pthread.h>
#include <sanitizer/msan_interface.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <ucontext.h>
#include <unistd.h>

void handler(int sig, siginfo_t *info, void *uctx) {
  __msan_check_mem_is_initialized(uctx, sizeof(ucontext_t));
#if defined(__GLIBC__) && defined(__x86_64__)
  auto *mctx = &static_cast<ucontext_t *>(uctx)->uc_mcontext;
  if (auto *fpregs = mctx->fpregs) {
    // The member names differ across header versions, but the actual layout
    // is always the same.  So avoid using members, just use arithmetic.
    const uint32_t *after_xmm =
        reinterpret_cast<const uint32_t *>(fpregs + 1) - 24;
    if (after_xmm[12] == FP_XSTATE_MAGIC1) {
      auto *xstate = reinterpret_cast<_xstate *>(mctx->fpregs);
      __msan_check_mem_is_initialized(xstate, sizeof(*xstate));
    }
  }
#endif
}

__attribute__((noinline)) void poison_stack() {
  char buf[64 << 10];
  printf("buf: %p-%p\n", buf, buf + sizeof(buf));
}

int main(int argc, char **argv) {
  struct sigaction act = {};
  act.sa_sigaction = handler;
  act.sa_flags = SA_SIGINFO;
  sigaction(SIGPROF, &act, 0);
  poison_stack();
  pthread_kill(pthread_self(), SIGPROF);
  return 0;
}

// CHECK-NOT: WARNING: MemorySanitizer:
