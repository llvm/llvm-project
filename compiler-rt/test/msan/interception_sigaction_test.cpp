// RUN: %clangxx_msan -O0 -g %s -o %t
// RUN: env MSAN_OPTIONS=handle_segv=2 %t 2>&1 | FileCheck %s
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <signal.h>
#include <string.h>

extern "C" int __interceptor_sigaction(int signum, const struct sigaction *act, struct sigaction *oldact);
extern "C" int sigaction(int signum, const struct sigaction *act, struct sigaction *oldact) {
  write(2, "sigaction call\n", sizeof("sigaction call\n") - 1);
  return __interceptor_sigaction(signum, act, oldact);
}

int main() {
  struct sigaction oldact;
  sigaction(SIGSEGV, nullptr, &oldact);

  if (oldact.sa_handler || oldact.sa_sigaction) {
    fprintf(stderr, "oldact filled\n");
  }
  return 0;
  // CHECK: sigaction call
  // CHECK: oldact filled
}
