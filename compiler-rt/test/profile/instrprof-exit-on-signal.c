// RUN: %clang_profgen -o %t %s
// RUN: %run LLVM_PROFILE_FILE="%15x%t.profraw" %t
// RUN: llvm-profdata show --all-functions %t.profraw | FileCheck %s
// CHECK: Total functions: 1

#include <signal.h>
#include <unistd.h>

int main() {
  kill(getpid(), SIGTERM);
  
  while (1) {
    /* loop forever */
  }
  return 1;
}
