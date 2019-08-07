// RUN: %clangxx_asan -O0 %s -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -O3 %s -o %t && not %run %t 2>&1 | FileCheck %s

// UNSUPPORTED: darwin

#include <assert.h>
#include <sys/wait.h>
#include <unistd.h>
#include <signal.h>

int main(int argc, char **argv) {
  pid_t pid = fork();
  if (pid) { // parent
    int x[3];
    int *status = x + argc * 3;
    int res;

    siginfo_t *si = (siginfo_t*)(x + argc * 3);
    res = waitid(P_ALL, 0, si, WEXITED | WNOHANG);
    // CHECK: stack-buffer-overflow
    // CHECK: {{WRITE of size .* at 0x.* thread T0}}
    // CHECK: {{in .*waitid}}
    // CHECK: {{in main .*waitid.cpp:}}
    // CHECK: is located in stack of thread T0 at offset
    // CHECK: {{in main}}
    return res != -1;
  }
  // child
  return 0;
}
