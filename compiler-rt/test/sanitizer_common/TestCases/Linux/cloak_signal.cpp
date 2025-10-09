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

void my_signal_sighandler(int signum) {
  printf("Custom signal handler\n");
  exit(1);
}

int main(int argc, char *argv[]) {
  __sighandler_t old = signal(SIGSEGV, &my_signal_sighandler);
  if (old == SIG_DFL)
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
