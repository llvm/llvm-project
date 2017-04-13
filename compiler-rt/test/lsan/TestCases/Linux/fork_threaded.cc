// Test that thread local data is handled correctly after forking without
// exec(). In this test leak checking is initiated from a non-main thread.
// RUN: %clangxx_lsan %s -o %t
// RUN: %run %t 2>&1

#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>

__thread void *thread_local_var;

void *exit_thread_func(void *arg) {
  exit(0);
}

void ExitFromThread() {
  pthread_t tid;
  int res;
  res = pthread_create(&tid, 0, exit_thread_func, 0);
  assert(res == 0);
  pthread_join(tid, 0);
}

int main() {
  int status = 0;
  thread_local_var = malloc(1337);
  pid_t pid = fork();
  assert(pid >= 0);
  if (pid > 0) {
    waitpid(pid, &status, 0);
    assert(WIFEXITED(status));
    return WEXITSTATUS(status);
  } else {
    // Spawn a thread and call exit() from there, to check that we track main
    // thread's pid correctly even if leak checking is initiated from another
    // thread.
    ExitFromThread();
  }
  return 0;
}
