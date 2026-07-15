// UNSUPPORTED: android
// UNSUPPORTED: hwasan

// RUN: %clangxx -O0 %s -o %t

// Sanitizer signal handler not installed; custom signal handler installed
// RUN: %env_tool_opts=handle_segv=0:cloak_sanitizer_signal_handlers=false not %run %t 2>&1 | FileCheck %s --check-prefixes=DEFAULT,CUSTOM
// RUN: %env_tool_opts=handle_segv=0:cloak_sanitizer_signal_handlers=true not %run %t 2>&1 | FileCheck %s --check-prefixes=DEFAULT,CUSTOM

// Sanitizer signal handler installed but overriden by custom signal handler
// RUN: %env_tool_opts=handle_segv=1:cloak_sanitizer_signal_handlers=false not %run %t 2>&1 | FileCheck %s --check-prefixes=NONDEFAULT,CUSTOM
// RUN: %env_tool_opts=handle_segv=1:cloak_sanitizer_signal_handlers=true not %run %t 2>&1 | FileCheck %s --check-prefixes=DEFAULT,CUSTOM

// Sanitizer signal handler installed immutably
// N.B. for handle_segv=2 with cloaking off, there is a pre-existing difference
//      in signal vs. sigaction: signal effectively cloaks the handler.
// RUN: %env_tool_opts=handle_segv=2:cloak_sanitizer_signal_handlers=false not %run %t 2>&1 | FileCheck %s --check-prefixes=NONDEFAULT,SANITIZER
// RUN: %env_tool_opts=handle_segv=2:cloak_sanitizer_signal_handlers=true not %run %t 2>&1 | FileCheck %s --check-prefixes=DEFAULT,SANITIZER

#include <signal.h>
#include <stdio.h>
#include <stdlib.h>

void handler(int signum, siginfo_t *info, void *context) {
  printf("Custom signal handler\n");
  exit(1);
}

int main(int argc, char *argv[]) {
  struct sigaction sa = {0};
  struct sigaction old = {0};
  sa.sa_flags = SA_SIGINFO;
  sa.sa_sigaction = &handler;
  sigaction(SIGSEGV, &sa, &old);

  if (reinterpret_cast<void *>(old.sa_sigaction) == SIG_DFL)
    printf("Old handler: default\n");
  // DEFAULT: Old handler: default
  else
    printf("Old handler: non-default\n");
  // NONDEFAULT: Old handler: non-default

  fflush(stdout);

  // Trying to organically segfault by dereferencing a pointer can be tricky
  // in builds with assertions. Additionally, some older platforms may SIGBUS
  // instead.
  raise(SIGSEGV);
  // CUSTOM: Custom signal handler
  // SANITIZER: Sanitizer:DEADLYSIGNAL

  return 0;
}
