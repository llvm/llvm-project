// RUN: %clang_safestack %s -pthread -o %t
// RUN: %run %t 0
// RUN: not --crash %run %t 1

// Test unsafe stack deallocation. Unsafe stacks are not deallocated immediately
// at thread exit. They are deallocated by following exiting threads.

#include <stdlib.h>
#include <string.h>
#include <pthread.h>

enum { kBufferSize = (1 << 15) };

void *start(void *ptr)
{
  char buffer[kBufferSize];
  return buffer;
}

int main(int argc, char **argv)
{
  int arg = atoi(argv[1]);

  pthread_t t1, t2;
  char *t1_buffer = NULL;

  if (pthread_create(&t1, NULL, start, NULL))
    abort();
  if (pthread_join(t1, &t1_buffer))
    abort();

  // Stack has not yet been deallocated
  memset(t1_buffer, 0, kBufferSize);

  if (arg == 0)
    return 0;

  for (int i = 0; i < 3; i++) {
    if (pthread_create(&t2, NULL, start, NULL))
      abort();
    // Second thread destructor cleans up the first thread's stack.
    if (pthread_join(t2, NULL))
      abort();

    // Should segfault here
    memset(t1_buffer, 0, kBufferSize);

    // PR39001: Re-try in the rare case that pthread_join() returns before the
    // thread finishes exiting in the kernel--hence the tgkill() check for t1
    // returns that it's alive despite pthread_join() returning.
    sleep(1);
  }
  return 0;
}
