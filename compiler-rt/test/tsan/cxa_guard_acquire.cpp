// RUN: %clangxx_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s

#include "test.h"
#include <assert.h>
#include <sanitizer/tsan_interface.h>
#include <stdio.h>

// We only enter a potentially blocking region on thread contention. To reliably
// trigger this, we force the initialization function to block until another
// thread has entered the potentially blocking region.

static bool init_done = false;

namespace __tsan {

#if (__APPLE__)
__attribute__((weak))
#endif
void OnPotentiallyBlockingRegionBegin() {
  assert(!init_done);
  printf("Enter potentially blocking region\n");
  // Signal the other thread to finish initialization.
  barrier_wait(&barrier);
}

#if (__APPLE__)
__attribute__((weak))
#endif
void OnPotentiallyBlockingRegionEnd() {
  printf("Exit potentially blocking region\n");
}

} // namespace __tsan

struct LazyInit {
  LazyInit() {
    assert(!init_done);
    printf("Enter constructor\n");
    // Wait for the other thread to get to the blocking region.
    barrier_wait(&barrier);
    printf("Exit constructor\n");
  }
};

const LazyInit &get_lazy_init() {
  static const LazyInit lazy_init;
  return lazy_init;
}

void *thread(void *arg) {
  get_lazy_init();
  return nullptr;
}

struct LazyInit2 {
  LazyInit2() { printf("Enter constructor 2\n"); }
};

const LazyInit2 &get_lazy_init2() {
  static const LazyInit2 lazy_init2;
  return lazy_init2;
}

int main(int argc, char **argv) {
  // CHECK: Enter main
  printf("Enter main\n");

  // If initialization is contended, the blocked thread should enter a
  // potentially blocking region. Note that we use a DAG check because it is
  // possible for Thread 1 to acquire the guard, then Thread 2 fail to acquire
  // the guard then call `OnPotentiallyBlockingRegionBegin` and print "Enter
  // potentially blocking region\n", before Thread 1 manages to reach "Enter
  // constructor\n". This is exceptionally rare, but can be replicated by
  // inserting a `sleep(1)` between `LazyInit() {` and `printf("Enter
  // constructor\n");`. Due to the barrier it is not possible for the exit logs
  // to be inverted.
  //
  // CHECK-DAG: Enter constructor
  // CHECK-DAG: Enter potentially blocking region
  // CHECK-NEXT: Exit constructor
  // CHECK-NEXT: Exit potentially blocking region
  barrier_init(&barrier, 2);
  pthread_t th1, th2;
  pthread_create(&th1, nullptr, thread, nullptr);
  pthread_create(&th2, nullptr, thread, nullptr);
  pthread_join(th1, nullptr);
  pthread_join(th2, nullptr);

  // Now that the value has been initialized, subsequent calls should not enter
  // a potentially blocking region.
  init_done = true;
  get_lazy_init();

  // If uncontended, there is no potentially blocking region.
  //
  // CHECK-NEXT: Enter constructor 2
  get_lazy_init2();
  get_lazy_init2();

  // CHECK-NEXT: Exit main
  printf("Exit main\n");
  return 0;
}
