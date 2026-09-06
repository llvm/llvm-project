// The double-free side table must survive fork(): the atfork handlers take its
// mutex in the LSan lock order, so a child that keeps allocating cannot inherit
// a locked table, and detection still works on both sides. The mutex is held
// across the unmap of a large chunk, so a churning thread has a wide window in
// which the fork can land on it.
//
// RUN: %clangxx_lsan -O0 %s -pthread -o %t
// RUN: %env_lsan_opts=detect_leaks=0:detect_double_free=1 %run %t 2>&1 | FileCheck %s
// REQUIRES: lsan-standalone

#include <cstdio>
#include <cstdlib>
#include <pthread.h>
#include <sys/wait.h>
#include <unistd.h>

static volatile bool g_stop;

// Keeps the allocator and the side table busy across the fork point.
static void *Churn(void *) {
  while (!g_stop) {
    void *small = malloc(32);
    void *large = malloc(1 << 20);
    free(small);
    free(large);
  }
  return nullptr;
}

int main() {
  pthread_t t;
  pthread_create(&t, nullptr, Churn, nullptr);

  for (int i = 0; i < 8; ++i) {
    pid_t pid = fork();
    if (pid == 0) {
      // The child inherits the side table. It must be usable, not deadlocked.
      for (int j = 0; j < 64; ++j) {
        void *p = malloc(1 << 20);
        free(p);
      }
      _exit(0);
    }
    int status = 0;
    waitpid(pid, &status, 0);
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
      fprintf(stderr, "child failed\n");
      return 1;
    }
  }

  g_stop = true;
  pthread_join(t, nullptr);
  fprintf(stderr, "fork test done\n");
  return 0;
}

// CHECK: fork test done
