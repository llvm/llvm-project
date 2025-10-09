// XFAIL: msan
// XFAIL: tsan

// UNSUPPORTED: android
// UNSUPPORTED: hwasan

// RUN: %clangxx -O0 %s -o %t

// RUN: %env_tool_opts=handle_segv=1:cloak_sanitizer_signal_handlers=false not %run %t 2>&1 | FileCheck %s --check-prefix=UNCLOAKED
// RUN: %env_tool_opts=handle_segv=1:cloak_sanitizer_signal_handlers=true not %run %t 2>&1 | FileCheck %s --check-prefix=CLOAKED

#include <sanitizer/common_interface_defs.h>
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
  // CLOAKED: Old handler: default
  else
    printf("Old handler: non-default\n");
  // UNCLOAKED: Old handler: non-default

  char *c = (char *)0x123;
  printf("%d\n", *c);
  // UNCLOAKED,CLOAKED:Custom signal handler

  return 0;
}
