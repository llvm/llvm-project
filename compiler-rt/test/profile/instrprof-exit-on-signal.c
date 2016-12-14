// RUN: %clang_profgen -o %t %s
//
// Verify SIGTERM handling.
// RUN: %run LLVM_PROFILE_FILE="%15x%t.profraw" %t 15
// RUN: llvm-profdata show %t.profraw | FileCheck %s
//
// Verify SIGUSR1 handling.
// RUN: %run LLVM_PROFILE_FILE="%30x%t.profraw" %t 30
// RUN: llvm-profdata show %t.profraw | FileCheck %s

#include <stdlib.h>
#include <signal.h>
#include <unistd.h>

// CHECK: Total functions: 1
int main(int argc, char **argv) {
  (void)argc;

  int sig = atoi(argv[1]);
  kill(getpid(), sig);
  
  while (1) {
    /* loop forever */
  }
  return 1;
}
