// Check that init-order checking is properly disabled if pthread_create is
// called.

// RUN: %clangxx_asan -c -DCONFIG1 %s -o %t1.o
// RUN: %clangxx_asan -c           %s -o %t2.o
// RUN: %clangxx_asan -pthread %t1.o %t2.o -o %t
// RUN: %env_asan_opts=strict_init_order=true %run %t

#ifdef CONFIG1

#include <stdio.h>
#include <pthread.h>
#include <unistd.h>

void *bar(void *input, bool sleep_before_init) {
  if (sleep_before_init)
    usleep(500000);
  return input;
}

void *glob = bar((void*)0x1234, false);
extern void *glob2;

void *poll(void *arg) {
  void **glob = (void**)arg;
  while (true) {
    usleep(100000);
    printf("glob is now: %p\n", *glob);
  }
}

struct GlobalPollerStarter {
  GlobalPollerStarter() {
    pthread_t p;
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
    pthread_create(&p, 0, poll, &glob);
    pthread_attr_destroy(&attr);
    printf("glob poller is started");
  }
} global_poller;

int main() {
  printf("%p %p\n", glob, glob2);
  return 0;
}

#else // CONFIG1

void *bar(void *input, bool sleep_before_init);
void *glob2 = bar((void*)0x2345, true);

#endif
