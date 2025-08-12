// Regression test 1:
// When the stack size is 1<<16, SizeRequiredForFlags(16) == 2KB. This forces
// FakeStack's GetFrame() out of alignment if the FakeStack isn't padded.
// RUN: %clangxx_asan -pthread -fsanitize-address-use-after-return=always -O0 -DALIGNMENT=4096  -DTHREAD_COUNT=1 -DTHREAD_STACK_SIZE=65536 %s -o %t && %run %t 2>&1

// Regression test 2:
// Check that the FakeStack frame is aligned, beyond the typical 4KB page
// alignment. Alignment can happen by chance, so try this on many threads.
// RUN: %clangxx_asan -pthread -fsanitize-address-use-after-return=always -O0 -DALIGNMENT=8192  -DTHREAD_COUNT=32 -DTHREAD_STACK_SIZE=131072 %s -o %t && %run %t 2>&1
// RUN: %clangxx_asan -pthread -fsanitize-address-use-after-return=always -O0 -DALIGNMENT=16384 -DTHREAD_COUNT=32 -DTHREAD_STACK_SIZE=131072 %s -o %t && %run %t 2>&1

// Extra tests:
// RUN: %clangxx_asan -pthread -fsanitize-address-use-after-return=always -O0 -DALIGNMENT=4096  -DTHREAD_COUNT=32 -DTHREAD_STACK_SIZE=65536 %s -o %t && %run %t 2>&1
// RUN: %clangxx_asan -pthread -fsanitize-address-use-after-return=always -O0 -DALIGNMENT=8192  -DTHREAD_COUNT=32 -DTHREAD_STACK_SIZE=65536 %s -o %t && %run %t 2>&1
// RUN: %clangxx_asan -pthread -fsanitize-address-use-after-return=always -O0 -DALIGNMENT=16384 -DTHREAD_COUNT=32 -DTHREAD_STACK_SIZE=65536 %s -o %t && %run %t 2>&1
// RUN: %clangxx_asan -pthread -fsanitize-address-use-after-return=always -O0 -DALIGNMENT=4096  -DTHREAD_COUNT=32 -DTHREAD_STACK_SIZE=131072 %s -o %t && %run %t 2>&1
// RUN: %clangxx_asan -pthread -fsanitize-address-use-after-return=always -O0 -DALIGNMENT=8192  -DTHREAD_COUNT=32 -DTHREAD_STACK_SIZE=131072 %s -o %t && %run %t 2>&1
// RUN: %clangxx_asan -pthread -fsanitize-address-use-after-return=always -O0 -DALIGNMENT=16384 -DTHREAD_COUNT=32 -DTHREAD_STACK_SIZE=131072 %s -o %t && %run %t 2>&1

#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct alignas(ALIGNMENT) big_object {
  int x;
};

bool misaligned = false;

// Check whether the FakeStack frame is sufficiently aligned. Alignment can
// happen by chance, so try this on many threads.
void *Thread(void *unused) {
  big_object x;
  uint alignment = (unsigned long)&x % alignof(big_object);

  if (alignment != 0)
    misaligned = true;

  return nullptr;
}

int main(int argc, char **argv) {
  pthread_attr_t attr;
  pthread_attr_init(&attr);
#ifdef THREAD_STACK_SIZE
  pthread_attr_setstacksize(&attr, THREAD_STACK_SIZE);
#endif

  pthread_t threads[THREAD_COUNT];
  for (pthread_t &t : threads)
    pthread_create(&t, &attr, Thread, 0);

  pthread_attr_destroy(&attr);

  for (pthread_t &t : threads)
    pthread_join(t, 0);

  if (misaligned) {
    printf("Test failed: not perfectly aligned\n");
    exit(1);
  }

  return 0;
}
