// RUN: %clang_safestack %s -pthread -o %t
// RUN: %run %t

// Test that pthreads receive their own unsafe stack.

#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include "utils.h"

static int ptr_test = 42;

void *t1_start(void *ptr)
{
  if (ptr != &ptr_test)
    abort();

  // safe stack
  int val = ptr_test * 5;

  // unsafe stack
  char buffer[8096]; // two pages
  memset(buffer, val, sizeof (buffer));
  break_optimization(buffer);

  return ptr;
}

int main(int argc, char **argv)
{
  pthread_t t1;
  void *ptr = NULL;
  if (pthread_create(&t1, NULL, t1_start, &ptr_test))
    abort();
  if (pthread_join(t1, &ptr))
    abort();
  if (ptr != &ptr_test)
    abort();
  return 0;
}
