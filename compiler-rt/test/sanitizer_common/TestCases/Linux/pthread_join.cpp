// RUN: %clangxx -pthread %s -o %t
// RUN: %run %t 0

// FIXME: Crashes on some bots in pthread_exit.
// RUN: %run %t %if tsan %{ 0 %} %else %{ 1 %}

// REQUIRES: glibc

#include <assert.h>
#include <ctime>
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>

bool use_exit;
static void *fn(void *args) {
  sleep(1);
  auto ret = (void *)(~(uintptr_t)args);
  if (use_exit)
    pthread_exit(ret);
  return ret;
}

int main(int argc, char **argv) {
  use_exit = atoi(argv[1]);
  pthread_t thread[5];
  assert(!pthread_create(&thread[0], nullptr, fn, (void *)1000));
  assert(!pthread_create(&thread[1], nullptr, fn, (void *)1001));
  assert(!pthread_create(&thread[2], nullptr, fn, (void *)1002));
  assert(!pthread_create(&thread[3], nullptr, fn, (void *)1003));
  pthread_attr_t attr;
  assert(!pthread_attr_init(&attr));
  assert(!pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED));
  assert(!pthread_create(&thread[4], &attr, fn, (void *)1004));
  assert(!pthread_attr_destroy(&attr));

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
