// RUN: %clangxx -O1 %s -o %t && %env_tool_opts=handle_sigtrap=1 %run %t 2>&1 | FileCheck %s

// __builtin_debugtrap() does not raise SIGTRAP on these platforms.
// UNSUPPORTED: s390

#include <assert.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>

int in_handler;

void handler(int signo, siginfo_t *info, void *uctx) {
  fprintf(stderr, "in_handler: %d\n", in_handler);
  fflush(stderr);
  // CHECK: in_handler: 1
  _Exit(0);
}

int main() {
  struct sigaction a = {}, old = {};
  a.sa_sigaction = handler;
  a.sa_flags = SA_SIGINFO;
  sigaction(SIGTRAP, &a, &old);

  a = {};
  sigaction(SIGTRAP, 0, &a);
  assert(a.sa_sigaction == handler);
  assert(a.sa_flags & SA_SIGINFO);

  in_handler = 1;
  // Check that signal handler is not postponed by sanitizer.
  // Don't use raise here as it calls any signal handler immediately.
  __builtin_debugtrap();
  in_handler = 0;

  fprintf(stderr, "UNREACHABLE\n");
  return 1;
}
