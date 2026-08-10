// RUN: %clang_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s
#include "test.h"
#include <signal.h>
#include <unistd.h>
#include <errno.h>
#include <semaphore.h>

// Test that signals can be delivered to blocked pthread_cond_wait.
// https://github.com/google/sanitizers/issues/498

int g_thread_run = 1;
pthread_mutex_t mutex;
pthread_cond_t cond;

void sig_handler(int sig) {
  (void)sig;
  write(2, "SIGNAL\n", sizeof("SIGNAL\n") - 1);
  barrier_wait(&barrier);
}

void* my_thread(void* arg) {
  pthread_mutex_lock(&mutex);
  while (g_thread_run)
    pthread_cond_wait(&cond, &mutex);
  pthread_mutex_unlock(&mutex);
  return 0;
}

int main() {
  barrier_init(&barrier, 2);

  pthread_mutex_init(&mutex, 0);
  pthread_cond_init(&cond, 0);

  signal(SIGUSR1, &sig_handler);
  pthread_t thr;
  pthread_create(&thr, 0, &my_thread, 0);
  // wait for thread to get inside pthread_cond_wait
  // (can't use barrier_wait for that)
  sleep(1);
  pthread_kill(thr, SIGUSR1);
  barrier_wait(&barrier);
  pthread_mutex_lock(&mutex);
  g_thread_run = 0;
  pthread_cond_signal(&cond);
  pthread_mutex_unlock(&mutex);
  pthread_join(thr, 0);
  fprintf(stderr, "DONE\n");
  return 0;
}

// CHECK: SIGNAL
// CHECK: DONE
