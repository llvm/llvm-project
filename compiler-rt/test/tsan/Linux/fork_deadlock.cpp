// RUN: %clangxx_tsan -O1 %s -o %t && %env_tsan_opts=atexit_sleep_ms=0 %run %t 2>&1 | FileCheck %s

// This test models what happens on Mac when fork
// calls malloc/free inside of our atfork callbacks.
// and ensures that we don't deadlock on malloc/free calls.

#include "../test.h"
#include "syscall.h"
#include <errno.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>

void alloc_free_blocks() {
  // Allocate a bunch of blocks to drain local allocator cache
  // and provoke it to lock allocator global mutexes.
  const int kBlocks = 1000;
  void *blocks[kBlocks];
  for (int i = 0; i < kBlocks; i++) {
    void *p = malloc(10);
    *(volatile char *)p = 0;
    blocks[i] = p;
  }
  for (int i = 0; i < kBlocks; i++)
    free(blocks[i]);
}

__attribute__((disable_sanitizer_instrumentation)) extern "C" void
__tsan_test_only_on_fork() {
  const char *msg = "__tsan_test_only_on_fork\n";
  write(2, msg, strlen(msg));
  alloc_free_blocks();
}

static void *background(void *p) {
  for (;;)
    alloc_free_blocks();
  return 0;
}

int main() {
  pthread_t th;
  pthread_create(&th, 0, background, 0);
  pthread_detach(th);
  for (int i = 0; i < 10; i++) {
    int pid = myfork();
    if (pid < 0) {
      fprintf(stderr, "failed to fork (%d)\n", errno);
      exit(1);
    }
    if (pid == 0) {
      // child
      exit(0);
    }
    // parent
    while (wait(0) < 0) {
    }
  }
  fprintf(stderr, "DONE\n");
}

// CHECK: __tsan_test_only_on_fork
// CHECK: DONE
