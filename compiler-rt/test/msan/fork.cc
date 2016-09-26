// Test that chained origins are fork-safe.
// Run a number of threads that create new chained origins, then fork
// and verify that origin reads do not deadlock in the child process.

// RUN: %clangxx_msan -std=c++11 -fsanitize-memory-track-origins=2 -g -O3 %s -o %t
// RUN: MSAN_OPTIONS=store_context_size=1000,origin_history_size=0,origin_history_per_stack_limit=0 %run %t |& FileCheck %s

// Fun fact: if test output is redirected to a file (as opposed to
// being piped directly to FileCheck), we may lose some "done"s due to
// a kernel bug:
// https://lkml.org/lkml/2014/2/17/324

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

#include <sanitizer/msan_interface.h>

int done;

void copy_uninit_thread2() {
  volatile int x;
  volatile int v;
  while (true) {
    v = x;
    x = v;
    if (__atomic_load_n(&done, __ATOMIC_RELAXED))
      return;
  }
}

void copy_uninit_thread1(int level) {
  if (!level)
    copy_uninit_thread2();
  else
    copy_uninit_thread1(level - 1);
}

void *copy_uninit_thread(void *id) {
  copy_uninit_thread1((long)id);
  return 0;
}

// Run through stackdepot in the child process.
// If any of the hash table cells are locked, this may deadlock.
void child() {
  volatile int x;
  volatile int v;
  for (int i = 0; i < 10000; ++i) {
    v = x;
    x = v;
  }
  write(2, "done\n", 5);
}

void test() {
  const int kThreads = 10;
  pthread_t t[kThreads];
  for (int i = 0; i < kThreads; ++i)
    pthread_create(&t[i], NULL, copy_uninit_thread, (void*)(long)i);
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
  const int kChildren = 20;
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

// Expect 20 (== kChildren) "done" messages.
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
