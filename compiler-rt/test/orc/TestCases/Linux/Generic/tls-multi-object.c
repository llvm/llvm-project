// RUN: split-file %s %t
// RUN: %clang -fPIC -c %t/main.c -o %t/main.o
// RUN: %clang -fPIC -c %t/access.c -o %t/access.o
// RUN: %llvm_jitlink %t/main.o %t/access.o
// RUN: %llvm_jitlink -num-threads=2 %t/main.o %t/access.o
// RUN: %llvm_jitlink -slab-allocate=1MB %t/main.o %t/access.o
// REQUIRES: system-linux, host-arch-compatible
// REQUIRES: host-jit-triple={{(x86_64|aarch64)-.*}}

// Accesses from different objects must share one TLS instance per thread.
// Use canonical TLS sections to test key reuse independently of subsection
// registration.

//--- main.c
#include <pthread.h>

_Thread_local int Data = 7;
_Thread_local int Zero;

int *getData(void);
int *getZero(void);
void increment(void);

static int check(void) {
  if (&Data != getData() || &Zero != getZero())
    return 1;
  if (Data != 7 || Zero != 0)
    return 2;
  ++Data;
  ++Zero;
  increment();
  if (Data != 9 || Zero != 2 || *getData() != 9 || *getZero() != 2)
    return 3;
  return 0;
}

static void *worker(void *Arg) {
  *(int *)Arg = check();
  return 0;
}

int main(void) {
  if (check())
    return 1;
  pthread_t Thread;
  int Result = -1;
  if (pthread_create(&Thread, 0, worker, &Result))
    return 2;
  if (pthread_join(Thread, 0) || Result)
    return 3;
  // The worker must have had its own TLS instance.
  return Data != 9 || Zero != 2;
}

//--- access.c
extern _Thread_local int Data;
extern _Thread_local int Zero;

int *getData(void) { return &Data; }
int *getZero(void) { return &Zero; }
void increment(void) {
  ++Data;
  ++Zero;
}
