// RUN: %clangxx_tsan -O1 %s -o %t && %deflake %run %t | FileCheck %s
#include "test.h"

#if defined(__linux__)
#define USE_PTHREAD_SETNAME_NP __GLIBC_PREREQ(2, 12)
#define tsan_pthread_setname_np pthread_setname_np
#elif defined(__FreeBSD__)
#include <pthread_np.h>
#define USE_PTHREAD_SETNAME_NP 1
#define tasn_pthread_setname_np pthread_set_name_np
#elif defined(__NetBSD__)
#define USE_PTHREAD_SETNAME_NP 1
#define tsan_pthread_setname_np(a, b) pthread_setname_np((a), "%s", (void *)(b))
#else
#define USE_PTHREAD_SETNAME_NP 0
#endif

extern "C" void AnnotateThreadName(const char *f, int l, const char *name);

int Global;

void *Thread1(void *x) {
  barrier_wait(&barrier);
  AnnotateThreadName(__FILE__, __LINE__, "Thread1");
  Global++;
  return NULL;
}

void *Thread2(void *x) {
#if USE_PTHREAD_SETNAME_NP
  tsan_pthread_setname_np(pthread_self(), "Thread2");
#else
  AnnotateThreadName(__FILE__, __LINE__, "Thread2");
#endif
  Global--;
  barrier_wait(&barrier);
  return NULL;
}

int main() {
  barrier_init(&barrier, 2);
  pthread_t t[2];
  pthread_create(&t[0], NULL, Thread1, NULL);
  pthread_create(&t[1], NULL, Thread2, NULL);
  pthread_join(t[0], NULL);
  pthread_join(t[1], NULL);
}

// CHECK: WARNING: ThreadSanitizer: data race
// CHECK:   Thread T1 'Thread1'
// CHECK:   Thread T2 'Thread2'
