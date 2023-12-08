// RUN: %clangxx -std=c++11 %s -o %t
// RUN: %env_asan_opts=handle_segv=1 LD_PRELOAD=%shared_libasan not %run %t 2>&1 | FileCheck %s
// RUN: %env_asan_opts=handle_segv=2 LD_PRELOAD=%shared_libasan not %run %t 2>&1 | FileCheck %s

// RUN: %clangxx -std=c++11 -DTEST_INSTALL_SIG_HANDLER %s -o %t
// RUN: %env_asan_opts=handle_segv=0 LD_PRELOAD=%shared_libasan not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-HANDLER
// RUN: %env_asan_opts=handle_segv=1 LD_PRELOAD=%shared_libasan not %run %t 2>&1 | FileCheck %s
// RUN: %env_asan_opts=handle_segv=2 LD_PRELOAD=%shared_libasan not %run %t 2>&1 | FileCheck %s

// RUN: %clangxx -std=c++11 -DTEST_INSTALL_SIG_ACTION %s -o %t
// RUN: %env_asan_opts=handle_segv=0 LD_PRELOAD=%shared_libasan not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-ACTION
// RUN: %env_asan_opts=handle_segv=1 LD_PRELOAD=%shared_libasan not %run %t 2>&1 | FileCheck %s
// RUN: %env_asan_opts=handle_segv=2 LD_PRELOAD=%shared_libasan not %run %t 2>&1 | FileCheck %s

// REQUIRES: asan-dynamic-runtime

// This way of setting LD_PRELOAD does not work with Android test runner.
// REQUIRES: !android

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

#if defined(__mips__)
  unsigned long flags;
  __sighandler_t handler;
#else
  __sighandler_t handler;
  unsigned long flags;
#endif
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

struct KernelSigaction pre_asan = {};

static void Init() {
  int res = InternalSigaction(SIGSEGV, nullptr, &pre_asan);
  assert(res >= 0);
  assert(pre_asan.handler == SIG_DFL || pre_asan.handler == SIG_IGN);
#if defined(TEST_INSTALL_SIG_HANDLER)
  pre_asan = {};
  pre_asan.handler = &SigHandler;
  res = InternalSigaction(SIGSEGV, &pre_asan, nullptr);
  assert(res >= 0);
#elif defined(TEST_INSTALL_SIG_ACTION)
  pre_asan = {};
  pre_asan.flags = SA_SIGINFO | SA_NODEFER;
  pre_asan.handler = (__sighandler_t)&SigAction;
  res = InternalSigaction(SIGSEGV, &pre_asan, nullptr);
  assert(res >= 0);
#endif
}

__attribute__((section(".preinit_array"), used))
void (*__local_test_preinit)(void) = Init;

bool ExpectUserHandler() {
#if defined(TEST_INSTALL_SIG_HANDLER) || defined(TEST_INSTALL_SIG_ACTION)
  return !strcmp(getenv("ASAN_OPTIONS"), "handle_segv=0");
#endif
  return false;
}

int main(int argc, char *argv[]) {
  KernelSigaction post_asan = {};
  InternalSigaction(SIGSEGV, nullptr, &post_asan);

  assert(post_asan.handler != SIG_DFL);
  assert(post_asan.handler != SIG_IGN);
  assert(ExpectUserHandler() ==
         (post_asan.handler == pre_asan.handler));

  raise(SIGSEGV);
  printf("%s\n", handler);
  return 1;
}

// CHECK-NOT: TestSig
// CHECK: AddressSanitizer:DEADLYSIGNAL

// CHECK-HANDLER-NOT: AddressSanitizer:DEADLYSIGNAL
// CHECK-HANDLER: TestSigHandler

// CHECK-ACTION-NOT: AddressSanitizer:DEADLYSIGNAL
// CHECK-ACTION: TestSigAction
