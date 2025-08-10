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

struct alignas(16384) larry {
  int x;
};

bool misaligned = false;

void grandchild(void) {
  larry l2;
  uint alignment = (unsigned long)&l2 % alignof(larry);
  if (alignment != 0)
    misaligned = true;

  printf("Grandchild: address modulo alignment %u\n", alignment);
}

// Even if the FakeStack frame is aligned by chance to 16384, we can use an
// intervening stack frame to knock it out of alignment.
void child(void) {
  page p1;
  uint alignment = (unsigned long)&p1 % alignof(page);
  printf("Child: address modulo alignment is %u\n", alignment);
  if (alignment != 0)
    misaligned = true;

  grandchild();
}

// Check whether the FakeStack frame is sufficiently aligned. Alignment can
// happen by chance, so try this on many threads if you don't want
void *Thread(void *unused) {
  larry l1;
  uint alignment = (unsigned long)&l1 % alignof(larry);
  printf("Thread: address modulo alignment is %u\n", alignment);
  if (alignment != 0)
    misaligned = true;

  child();

  return NULL;
}

int main(int argc, char **argv) {
  pthread_attr_t attr;
  pthread_attr_init(&attr);

  pthread_t t[10];
  for (int i = 0; i < 10; i++) {
    pthread_create(&t[i], &attr, Thread, 0);
  }
  pthread_attr_destroy(&attr);
  for (int i = 0; i < 10; i++) {
    pthread_join(t[i], 0);
  }

  if (misaligned) {
    printf("Test failed: not perfectly aligned\n");
    exit(1);
  }

  return 0;
}
