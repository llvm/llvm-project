// ObjC runtime uses locks in a manner that is opaque to TSan; this test
// checks that the deadlock detector does not report any false positives
// as a result

// RUN: %clangxx_tsan %s -o %t -framework Foundation -fobjc-arc
// RUN: %env_tsan_opts=detect_deadlocks=1:ignore_noninstrumented_modules=1 %run %t 2>&1 | FileCheck %s

#import <Foundation/Foundation.h>
#include <pthread.h>

static NSObject *objs[64];

static void *worker(void *arg) {
  @autoreleasepool {
    unsigned seed = (unsigned)(uintptr_t)arg;
    for (int i = 0; i < 100000; i++) {
      // Always lock the lower index first, so there is no real inversion.
      int a = rand_r(&seed) % 63, b = a + 1 + rand_r(&seed) % (63 - a);
      @synchronized(objs[a]) {
        @synchronized(objs[b]) {
        }
      }
    }
  }
  return nullptr;
}

int main() {
  for (auto &o : objs)
    o = [NSObject new];
  pthread_t threads[2];
  for (int i = 0; i < 2; i++)
    pthread_create(&threads[i], nullptr, worker, (void *)(uintptr_t)(i + 1));
  for (auto &t : threads)
    pthread_join(t, nullptr);
  fprintf(stderr, "Done\n");
}

// CHECK-NOT: ThreadSanitizer
// CHECK: Done
