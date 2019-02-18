// RUN: %clang_safestack %s -pthread -o %t
// RUN: %run %t

// Test unsafe stack deallocation with custom stack sizes, in particular ensure
// that we correctly deallocate small stacks and don't accidentally deallocate
// adjacent memory.

#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

volatile int step = 0;

void *wait_until(void *ptr) {
  while ((int)ptr != step)
    usleep(1000);

  volatile char buf[64];
  buf[0] = 0;

  return NULL;
}

int main(int argc, char **argv) {
  pthread_t t1, t2, t3;

  pthread_attr_t small_stack_attr;
  pthread_attr_init(&small_stack_attr);
  pthread_attr_setstacksize(&small_stack_attr, 65536);

  if (pthread_create(&t3, NULL, wait_until, (void *)3))
    abort();
  if (pthread_create(&t1, &small_stack_attr, wait_until, (void *)1))
    abort();
  if (pthread_create(&t2, NULL, wait_until, (void *)2))
    abort();

  step = 1;
  if (pthread_join(t1, NULL))
    abort();

  step = 2;
  if (pthread_join(t2, NULL))
    abort();

  step = 3;
  if (pthread_join(t3, NULL))
    abort();

  pthread_attr_destroy(&small_stack_attr);
  return 0;
}
