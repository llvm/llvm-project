// RUN: %clangxx -pthread %s -o %t && %run %t

// XFAIL: msan

// REQUIRES: glibc

#include <assert.h>
#include <ctime>
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>

static void *fn(void *args) {
  sleep(1);
  return (void *)(~(uintptr_t)args);
}

int main(int argc, char **argv) {
  pthread_t thread[4];
  assert(!pthread_create(&thread[0], 0, fn, (void *)1000));
  assert(!pthread_create(&thread[1], 0, fn, (void *)1001));
  assert(!pthread_create(&thread[2], 0, fn, (void *)1002));
  assert(!pthread_create(&thread[3], 0, fn, (void *)1003));

  assert(!pthread_detach(thread[0]));

  {
    void *res;
    while (pthread_tryjoin_np(thread[1], &res))
      sleep(1);
    assert(~(uintptr_t)res == 1001);
  }

  {
    void *res;
    timespec tm = {0, 1};
    while (pthread_timedjoin_np(thread[2], &res, &tm))
      sleep(1);
    assert(~(uintptr_t)res == 1002);
  }

  {
    void *res;
    assert(!pthread_join(thread[3], &res));
    assert(~(uintptr_t)res == 1003);
  }

  return 0;
}
