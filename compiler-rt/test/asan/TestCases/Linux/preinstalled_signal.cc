// clang-format off
// RUN: %clangxx -std=c++11 %s -o %t
// RUN: env LD_PRELOAD=%shared_libasan %env_asan_opts=handle_segv=1 not %run %t 2>&1 | FileCheck %s
// RUN: env LD_PRELOAD=%shared_libasan %env_asan_opts=handle_segv=2 not %run %t 2>&1 | FileCheck %s

// RUN: %clangxx -std=c++11 -DTEST_INSTALL_SIG_HANDLER %s -o %t
// RUN: env LD_PRELOAD=%shared_libasan %env_asan_opts=handle_segv=1 not %run %t 2>&1 | FileCheck --check-prefix=CHECK-HANDLER %s
// RUN: env LD_PRELOAD=%shared_libasan %env_asan_opts=handle_segv=2 not %run %t 2>&1 | FileCheck %s

// RUN: %clangxx -std=c++11 -DTEST_INSTALL_SIG_ACTION %s -o %t
// RUN: env LD_PRELOAD=%shared_libasan %env_asan_opts=handle_segv=1 not %run %t 2>&1 | FileCheck --check-prefix=CHECK-ACTION %s
// RUN: env LD_PRELOAD=%shared_libasan %env_asan_opts=handle_segv=2 not %run %t 2>&1 | FileCheck %s

// REQUIRES: asan-dynamic-runtime

// This way of setting LD_PRELOAD does not work with Android test runner.
// REQUIRES: not-android
// clang-format on

#include <assert.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>

const char *handler = nullptr;
void SigHandler(int signum) { handler = "TestSigHandler"; }
void SigAction(int, siginfo_t *, void *) { handler = "TestSigAction"; }

struct KernelSigaction {
  __sighandler_t handler;
  unsigned long flags;
  void (*restorer)();
  char unused[1024];
};

#if defined(__x86_64__)
extern "C" void restorer();
asm("restorer:mov $15,%rax\nsyscall");
#endif

int InternalSigaction(int sig, KernelSigaction *act, KernelSigaction *oact) {
  if (act) {
#if defined(__x86_64__)
    act->flags |= 0x04000000;
    act->restorer = &restorer;
#endif
  }
  return syscall(__NR_rt_sigaction, sig, act, oact, NSIG / 8);
}

struct KernelSigaction sigact = {};

static void Init() {
  int res = InternalSigaction(SIGSEGV, nullptr, &sigact);
  assert(res >= 0);
  assert(sigact.handler == SIG_DFL || sigact.handler == SIG_IGN);
#if defined(TEST_INSTALL_SIG_HANDLER)
  sigact = {};
  sigact.handler = &SigHandler;
  res = InternalSigaction(SIGSEGV, &sigact, nullptr);
  assert(res >= 0);
#elif defined(TEST_INSTALL_SIG_ACTION)
  sigact = {};
  sigact.flags = SA_SIGINFO | SA_NODEFER;
  sigact.handler = (__sighandler_t)&SigAction;
  res = InternalSigaction(SIGSEGV, &sigact, nullptr);
  assert(res >= 0);
#endif
}

__attribute__((section(".preinit_array"), used))
void (*__local_test_preinit)(void) = Init;

bool ShouldAsanInstallHandlers() {
#if defined(TEST_INSTALL_SIG_HANDLER) || defined(TEST_INSTALL_SIG_ACTION)
  return !strcmp(getenv("ASAN_OPTIONS"), "handle_segv=2");
#endif
  return true;
}

int main(int argc, char *argv[]) {
  KernelSigaction sigact_asan = {};
  InternalSigaction(SIGSEGV, nullptr, &sigact_asan);

  assert(sigact_asan.handler != SIG_DFL);
  assert(sigact_asan.handler != SIG_IGN);
  assert(ShouldAsanInstallHandlers() ==
         (sigact_asan.handler != sigact.handler));

  raise(SIGSEGV);
  printf("%s\n", handler);
  return 1;
}

// CHECK-NOT: TestSig
// CHECK: ASAN:DEADLYSIGNAL

// CHECK-HANDLER-NOT: ASAN:DEADLYSIGNAL
// CHECK-HANDLER: TestSigHandler

// CHECK-ACTION-NOT: ASAN:DEADLYSIGNAL
// CHECK-ACTION: TestSigAction
