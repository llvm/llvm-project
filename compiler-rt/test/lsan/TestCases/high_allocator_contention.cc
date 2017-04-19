// A benchmark that executes malloc/free pairs in parallel.
// Usage: ./a.out number_of_threads total_number_of_allocations
// RUN: LSAN_BASE="detect_leaks=1:use_ld_allocations=0"
// RUN: %clangxx_lsan %s -o %t
// RUN: %env_lsan_opts=$LSAN_BASE %run %t 5 1000000 2>&1
#include <assert.h>
#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>

int num_threads;
int total_num_alloc;
const int kMaxNumThreads = 5000;
pthread_t tid[kMaxNumThreads];

pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
bool go = false;

void *thread_fun(void *arg) {
  pthread_mutex_lock(&mutex);
  while (!go) pthread_cond_wait(&cond, &mutex);
  pthread_mutex_unlock(&mutex);
  for (int i = 0; i < total_num_alloc / num_threads; i++) {
    void *p = malloc(10);
    __asm__ __volatile__("" : : "r"(p) : "memory");
    free((void *)p);
  }
  return 0;
}

int main(int argc, char** argv) {
  assert(argc == 3);
  num_threads = atoi(argv[1]);
  assert(num_threads > 0);
  assert(num_threads <= kMaxNumThreads);
  total_num_alloc = atoi(argv[2]);
  assert(total_num_alloc > 0);
  printf("%d threads, %d allocations in each\n", num_threads,
         total_num_alloc / num_threads);
  for (int i = 0; i < num_threads; i++)
    pthread_create(&tid[i], 0, thread_fun, 0);
  pthread_mutex_lock(&mutex);
  go = true;
  pthread_cond_broadcast(&cond);
  pthread_mutex_unlock(&mutex);
  for (int i = 0; i < num_threads; i++) pthread_join(tid[i], 0);
  return 0;
}
