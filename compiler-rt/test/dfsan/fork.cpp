// Test that chained origins are fork-safe.
// Run a number of threads that create new chained origins, then fork
// and verify that origin reads do not deadlock in the child process.
//
// RUN: %clangxx_dfsan %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s
//
// RUN: %clangxx_dfsan -mllvm -dfsan-track-origins=1 %s -o %t
// RUN: env DFSAN_OPTIONS=store_context_size=1000,origin_history_size=0,origin_history_per_stack_limit=0 %run %t 2>&1 | FileCheck %s

#include <assert.h>
#include <errno.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <sanitizer/dfsan_interface.h>

int done;

void copy_labels_thread2() {
  volatile int x = 0;
  volatile int v = 0;
  dfsan_set_label(8, (void *)&x, sizeof(x));
  while (true) {
    v = x;
    x = v;
    if (__atomic_load_n(&done, __ATOMIC_RELAXED))
      return;
  }
}

void copy_labels_thread1(int level) {
  if (!level)
    copy_labels_thread2();
  else
    copy_labels_thread1(level - 1);
}

void *copy_labels_thread(void *id) {
  copy_labels_thread1((long)id);
  return 0;
}

// Run through stackdepot in the child process.
// If any of the hash table cells are locked, this may deadlock.
void child() {
  volatile int x = 0;
  volatile int v = 0;
  dfsan_set_label(16, (void *)&x, sizeof(x));
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
    pthread_create(&t[i], NULL, copy_labels_thread, (void *)(long)i);
  usleep(100000);
  pid_t pid = fork();
  if (pid) {
    // parent
    __atomic_store_n(&done, 1, __ATOMIC_RELAXED);
    pid_t p;
    while ((p = wait(NULL)) == -1) {
    }
  } else {
    // child
    child();
  }
}

int main() {
  const int kChildren = 20;
  for (int i = 0; i < kChildren; ++i) {
    pid_t pid = fork();
    assert(dfsan_get_label(pid) == 0);
    if (pid) {
      // parent
    } else {
      test();
      exit(0);
    }
  }

  for (int i = 0; i < kChildren; ++i) {
    pid_t p;
    while ((p = wait(NULL)) == -1) {
    }
  }

  return 0;
}

// Expect 20 (== kChildren) "done" messages.
// CHECK-COUNT-20: done
