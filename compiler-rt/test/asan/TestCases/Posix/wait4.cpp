// RUN: %clangxx_asan -DWAIT4 -O0 %s -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -DWAIT4 -O3 %s -o %t && not %run %t 2>&1 | FileCheck %s

// RUN: %clangxx_asan -DWAIT4_RUSAGE -O0 %s -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -DWAIT4_RUSAGE -O3 %s -o %t && not %run %t 2>&1 | FileCheck %s

// XFAIL: android
// UNSUPPORTED: darwin

#include <assert.h>
#include <sys/wait.h>
#include <unistd.h>

int main(int argc, char **argv) {
  // This test passes on some versions of Android NDK and fails on other.
  // https://code.google.com/p/memory-sanitizer/issues/detail?id=64
  // Make it fail unconditionally on Android.
#ifdef __ANDROID__
  return 0;
#endif

  pid_t pid = fork();
  if (pid) { // parent
    int x[3];
    int *status = x + argc * 3;
    int res;
#if defined(WAIT4)
    res = wait4(pid, status, WNOHANG, NULL);
#elif defined(WAIT4_RUSAGE)
    struct rusage *ru = (struct rusage*)(x + argc * 3);
    int good_status;
    res = wait4(pid, &good_status, WNOHANG, ru);
#endif
    // CHECK: stack-buffer-overflow
    // CHECK: {{WRITE of size .* at 0x.* thread T0}}
    // CHECK: {{in .*wait}}
    // CHECK: {{in main .*wait4.cpp:}}
    // CHECK: is located in stack of thread T0 at offset
    // CHECK: {{in main}}
    return res == -1 ? 1 : 0;
  }
  // child
  return 0;
}
