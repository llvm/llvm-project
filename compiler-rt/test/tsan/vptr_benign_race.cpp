// RUN: %clangxx_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s
#include <pthread.h>
#include <stdio.h>

struct A {
  A() {
    pthread_mutex_init(&m, 0);
    pthread_cond_init(&c, 0);
    signaled = false;
  }
  virtual void F() {
  }
  void Done() {
    pthread_mutex_lock(&m);
    signaled = true;
    pthread_cond_signal(&c);
    pthread_mutex_unlock(&m);
  }
  virtual ~A() {
  }
  pthread_mutex_t m;
  pthread_cond_t c;
  bool signaled;
};

struct B : A {
  virtual void F() {
  }
  virtual ~B() {
    pthread_mutex_lock(&m);
    while (!signaled)
      pthread_cond_wait(&c, &m);
    pthread_mutex_unlock(&m);
  }
};

static A *obj = new B;

void *Thread1(void *x) {
  obj->F();
  obj->Done();
  return NULL;
}

void *Thread2(void *x) {
  delete obj;
  return NULL;
}

int main() {
  pthread_t t[2];
  pthread_create(&t[0], NULL, Thread1, NULL);
  pthread_create(&t[1], NULL, Thread2, NULL);
  pthread_join(t[0], NULL);
  pthread_join(t[1], NULL);
  fprintf(stderr, "PASS\n");
}
// CHECK: PASS
// CHECK-NOT: WARNING: ThreadSanitizer: data race
