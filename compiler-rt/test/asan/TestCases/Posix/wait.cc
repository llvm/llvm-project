// RUN: %clangxx_asan -DWAIT -O0 %s -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -DWAIT -O3 %s -o %t && not %run %t 2>&1 | FileCheck %s

// RUN: %clangxx_asan -DWAITPID -O0 %s -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -DWAITPID -O3 %s -o %t && not %run %t 2>&1 | FileCheck %s

// UNSUPPORTED: darwin

#include <assert.h>
#include <sys/wait.h>
#include <unistd.h>

int main(int argc, char **argv) {
  pid_t pid = fork();
  if (pid) { // parent
    int x[3];
    int *status = x + argc * 3;
    int res;
#if defined(WAIT)
    res = wait(status);
#elif defined(WAITPID)
    res = waitpid(pid, status, WNOHANG);
#endif
    // CHECK: stack-buffer-overflow
    // CHECK: {{WRITE of size .* at 0x.* thread T0}}
    // CHECK: {{in .*wait}}
    // CHECK: {{in main .*wait.cc:}}
    // CHECK: is located in stack of thread T0 at offset
    // CHECK: {{in main}}
    return res == -1 ? 1 : 0;
  }
  // child
  return 0;
}
