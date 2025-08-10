// RUN: %clangxx_asan -fsanitize-address-use-after-return=always -O0 %s -o %t && %run %t 2>&1
// XFAIL: *

#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct alignas(4096) page {
  int x;
};

void *Thread(void *unused) {
  page p1;
  uint alignment = (unsigned long)&p1 % alignof(page);
  printf("Thread: address modulo alignment is %u\n", alignment);
  assert(alignment == 0);

  return NULL;
}

int main(int argc, char **argv) {
  pthread_attr_t attr;
  pthread_attr_init(&attr);

  // When the stack size is 1<<16, FakeStack's GetFrame() is out of alignment,
  // because SizeRequiredForFlags(16) == 2K.
  pthread_attr_setstacksize(&attr, 1 << 16);

  pthread_t t;
  pthread_create(&t, &attr, Thread, 0);
  pthread_attr_destroy(&attr);
  pthread_join(t, 0);

  return 0;
}
