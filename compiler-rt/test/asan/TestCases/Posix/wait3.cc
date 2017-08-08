// RUN: %clangxx_asan -DWAIT3 -O0 %s -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -DWAIT3 -O3 %s -o %t && not %run %t 2>&1 | FileCheck %s

// RUN: %clangxx_asan -DWAIT3_RUSAGE -O0 %s -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -DWAIT3_RUSAGE -O3 %s -o %t && not %run %t 2>&1 | FileCheck %s

// UNSUPPORTED: android,darwin

#include <assert.h>
#include <sys/wait.h>
#include <unistd.h>

int main(int argc, char **argv) {
  pid_t pid = fork();
  if (pid) { // parent
    int x[3];
    int *status = x + argc * 3;
    int res;
#if defined(WAIT3)
    res = wait3(status, WNOHANG, NULL);
#elif defined(WAIT3_RUSAGE)
    struct rusage *ru = (struct rusage*)(x + argc * 3);
    int good_status;
    res = wait3(&good_status, WNOHANG, ru);
#endif
    // CHECK: stack-buffer-overflow
    // CHECK: {{WRITE of size .* at 0x.* thread T0}}
    // CHECK: {{in .*wait}}
    // CHECK: {{in main .*wait3.cc:}}
    // CHECK: is located in stack of thread T0 at offset
    // CHECK: {{in main}}
    return res == -1 ? 1 : 0;
  }
  // child
  return 0;
}
