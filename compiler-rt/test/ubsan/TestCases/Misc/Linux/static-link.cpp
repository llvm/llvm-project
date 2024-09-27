// REQUIRES: ubsan-standalone
// REQUIRES: target={{x86_64.*}}
// UNSUPPORTED: i386-target-arch, internal_symbolizer
// RUN: %clangxx -fsanitize=bool -static  %s -o %t && env UBSAN_OPTIONS=handle_segv=0:handle_sigbus=0:handle_sigfpe=0 %run %t 2>&1 | FileCheck %s
// RUN: %run %t 2>&1 | FileCheck %s
#include <signal.h>
#include <stdio.h>

int main() {
  struct sigaction old_action;
  sigaction(SIGINT, nullptr, &old_action);
  // CHECK: Warning: REAL(sigaction_symname) == nullptr.
  printf("PASS\n");
  // CHECK: PASS
}
