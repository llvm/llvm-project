// Regression test for
// https://code.google.com/p/address-sanitizer/issues/detail?id=180
// Fails with debug checks: https://bugs.llvm.org/show_bug.cgi?id=46860
// XFAIL: !compiler-rt-optimized && tsan

// FIXME: Implement.
// XFAIL: hwasan

// RUN: %clangxx -O0 %s -o %t

// RUN: %env_tool_opts=handle_segv=0 not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK0
// RUN: %env_tool_opts=handle_segv=1 not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK1
// RUN: %env_tool_opts=handle_segv=2 not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK2

// RUN: %env_tool_opts=handle_segv=0:allow_user_segv_handler=0 not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK0
// RUN: %env_tool_opts=handle_segv=1:allow_user_segv_handler=0 not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK2
// RUN: %env_tool_opts=handle_segv=2:allow_user_segv_handler=0 not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK2

// RUN: %env_tool_opts=handle_segv=0:allow_user_segv_handler=1 not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK0
// RUN: %env_tool_opts=handle_segv=1:allow_user_segv_handler=1 not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK1
// RUN: %env_tool_opts=handle_segv=2:allow_user_segv_handler=1 not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK2

// Flaky errors in debuggerd with "waitpid returned unexpected pid (0)" in logcat.
// UNSUPPORTED: android && i386-target-arch

#include <signal.h>
#include <stdio.h>
#include <stdlib.h>

struct sigaction original_sigaction_sigbus;
struct sigaction original_sigaction_sigsegv;

void User_OnSIGSEGV(int signum, siginfo_t *siginfo, void *context) {
  fprintf(stderr, "User sigaction called\n");
  struct sigaction original_sigaction = {};
  if (signum == SIGBUS)
    original_sigaction = original_sigaction_sigbus;
  else if (signum == SIGSEGV)
    original_sigaction = original_sigaction_sigsegv;
  else {
    printf("Invalid signum");
    exit(1);
  }
  if (original_sigaction.sa_flags | SA_SIGINFO) {
    if (original_sigaction.sa_sigaction)
      original_sigaction.sa_sigaction(signum, siginfo, context);
  } else {
    if (original_sigaction.sa_handler)
      original_sigaction.sa_handler(signum);
  }
  exit(1);
}

int DoSEGV() {
  volatile int *x = 0;
  return *x;
}

bool InstallHandler(int signum, struct sigaction *original_sigaction) {
  struct sigaction user_sigaction = {};
  user_sigaction.sa_sigaction = User_OnSIGSEGV;
  user_sigaction.sa_flags = SA_SIGINFO;
  if (sigaction(signum, &user_sigaction, original_sigaction)) {
    perror("sigaction");
    return false;
  }
  return true;
}

int main() {
  // Let's install handlers for both SIGSEGV and SIGBUS, since pre-Yosemite
  // 32-bit Darwin triggers SIGBUS instead.
  if (InstallHandler(SIGSEGV, &original_sigaction_sigsegv) &&
      InstallHandler(SIGBUS, &original_sigaction_sigbus)) {
    fprintf(stderr, "User sigaction installed\n");
  }
  return DoSEGV();
}

// CHECK0-NOT: Sanitizer:DEADLYSIGNAL
// CHECK0-NOT: Sanitizer: SEGV on unknown address
// CHECK0: User sigaction installed
// CHECK0-NEXT: User sigaction called

// CHECK1: User sigaction installed
// CHECK1-NEXT: User sigaction called
// CHECK1-NEXT: Sanitizer:DEADLYSIGNAL
// CHECK1: Sanitizer: SEGV on unknown address

// CHECK2-NOT: User sigaction called
// CHECK2: User sigaction installed
// CHECK2-NEXT: Sanitizer:DEADLYSIGNAL
// CHECK2: Sanitizer: SEGV on unknown address
