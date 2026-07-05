// Test that thread local data is handled correctly after forking without exec().
// RUN: %clangxx_lsan %s -o %t
// RUN: %run %t 2>&1

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>

__thread void *thread_local_var;

int main() {
  int status = 0;
  thread_local_var = malloc(1337);
  pid_t pid = fork();
  assert(pid >= 0);
  if (pid > 0) {
    waitpid(pid, &status, 0);
    assert(WIFEXITED(status));
    return WEXITSTATUS(status);
  }
  return 0;
}
