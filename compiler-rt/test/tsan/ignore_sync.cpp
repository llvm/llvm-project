// RUN: %clangxx_tsan -O1 %s -o %t && %deflake %run %t | FileCheck %s
#include <pthread.h>
#include <stdio.h>

extern "C" void AnnotateIgnoreSyncBegin(const char*, int);
extern "C" void AnnotateIgnoreSyncEnd(const char*, int);

int Global;
pthread_mutex_t Mutex = PTHREAD_MUTEX_INITIALIZER;

void *Thread(void *x) {
  AnnotateIgnoreSyncBegin(0, 0);
  pthread_mutex_lock(&Mutex);
  Global++;
  pthread_mutex_unlock(&Mutex);
  AnnotateIgnoreSyncEnd(0, 0);
  return 0;
}

int main() {
  pthread_t t;
  pthread_create(&t, 0, Thread, 0);
  pthread_mutex_lock(&Mutex);
  Global++;
  pthread_mutex_unlock(&Mutex);
  pthread_join(t, 0);
}

// CHECK: WARNING: ThreadSanitizer: data race

