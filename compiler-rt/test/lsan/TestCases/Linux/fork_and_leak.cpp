// Test that leaks detected after forking without exec().
// RUN: %clangxx_lsan %s -o %t && not %run %t 2>&1 | FileCheck %s

/// Fails on clang-cmake-aarch64-full (glibc 2.27-3ubuntu1.4).
// UNSUPPORTED: aarch64

#include <assert.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>

int main() {
  pid_t pid = fork();
  assert(pid >= 0);
  if (pid > 0) {
    int status = 0;
    waitpid(pid, &status, 0);
    assert(WIFEXITED(status));
    return WEXITSTATUS(status);
  } else {
    malloc(1337);
    // CHECK: LeakSanitizer: detected memory leaks
  }
  return 0;
}

