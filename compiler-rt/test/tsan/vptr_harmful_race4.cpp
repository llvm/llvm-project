// RUN: %clangxx_tsan -O1 %s -o %t && %deflake %run %t | FileCheck %s
#include "test.h"

struct A {
  virtual void F() {
  }

  virtual ~A() {
  }
};

struct B : A {
  virtual void F() {
  }
};

void *Thread(void *x) {
  barrier_wait(&barrier);
  ((A*)x)->F();
  return 0;
}

int main() {
  barrier_init(&barrier, 2);
  A *obj = new B;
  pthread_t t;
  pthread_create(&t, 0, Thread, obj);
  delete obj;
  barrier_wait(&barrier);
  pthread_join(t, 0);
}

// CHECK: WARNING: ThreadSanitizer: heap-use-after-free (virtual call vs free)

