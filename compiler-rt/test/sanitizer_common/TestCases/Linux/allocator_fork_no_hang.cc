// https://github.com/google/sanitizers/issues/774
// Test that sanitizer allocator is fork-safe.
// Run a number of threads that perform memory allocation/deallocation, then fork
// and verify that malloc/free do not deadlock in the child process.

// RUN: %clangxx -std=c++11 -O0 %s -o %t
// RUN: ASAN_OPTIONS=detect_leaks=0 %run %t 2>&1 | FileCheck %s

// Fun fact: if test output is redirected to a file (as opposed to
// being piped directly to FileCheck), we may lose some "done"s due to
// a kernel bug:
// https://lkml.org/lkml/2014/2/17/324

// UNSUPPORTED: tsan

// Flaky on PPC64.
// UNSUPPORTED: powerpc64-target-arch
// UNSUPPORTED: powerpc64le-target-arch

#include <pthread.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/time.h>
#include <signal.h>
#include <errno.h>

int done;

void *worker(void *arg) {
  while (true) {
    void *p = malloc(4);
    if (__atomic_load_n(&done, __ATOMIC_RELAXED))
      return 0;
  }
  return 0;
}

// Run through malloc/free in the child process.
// This can deadlock on allocator cache refilling.
void child() {
  for (int i = 0; i < 10000; ++i) {
    void *p = malloc(4);
  }
  write(2, "done\n", 5);
}

void test() {
  const int kThreads = 10;
  pthread_t t[kThreads];
  for (int i = 0; i < kThreads; ++i)
    pthread_create(&t[i], NULL, worker, (void*)(long)i);
  usleep(100000);
  pid_t pid = fork();
  if (pid) {
    // parent
    __atomic_store_n(&done, 1, __ATOMIC_RELAXED);
    pid_t p;
    while ((p = wait(NULL)) == -1) {  }
  } else {
    // child
    child();
  }
}

int main() {
  const int kChildren = 30;
  for (int i = 0; i < kChildren; ++i) {
    pid_t pid = fork();
    if (pid) {
      // parent
    } else {
      test();
      exit(0);
    }
  }

  for (int i = 0; i < kChildren; ++i) {
    pid_t p;
    while ((p = wait(NULL)) == -1) {  }
  }

  return 0;
}

// Expect 30 (== kChildren) "done" messages.
// CHECK: done
// CHECK: done
// CHECK: done
// CHECK: done
// CHECK: done
// CHECK: done
// CHECK: done
// CHECK: done
// CHECK: done
// CHECK: done
// CHECK: done
// CHECK: done
// CHECK: done
// CHECK: done
// CHECK: done
// CHECK: done
// CHECK: done
// CHECK: done
// CHECK: done
// CHECK: done
// CHECK: done
// CHECK: done
// CHECK: done
// CHECK: done
// CHECK: done
// CHECK: done
// CHECK: done
// CHECK: done
// CHECK: done
// CHECK: done
