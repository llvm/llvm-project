// Test that vfork() is fork().
// https://github.com/google/sanitizers/issues/925

// RUN: %clangxx_asan -O0 %s -o %t && %run %t 2>&1

#include <assert.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

int volatile global;

int main(int argc, char **argv) {
  pid_t pid = vfork();
  if (pid) {
    // parent
    int status;
    int res;
    do {
      res = waitpid(pid, &status, 0);
    } while (res >= 0 && !WIFEXITED(status) && !WIFSIGNALED(status));
    assert(global == 0);
  } else {
    // child
    global = 42;
    _exit(0);
  }

  return 0;
}
