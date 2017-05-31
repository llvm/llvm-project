// RUN: %clang_tsan %s -o %t -framework Foundation
// RUN: %env_tsan_opts=ignore_interceptors_accesses=1 %run %t 2>&1 | FileCheck %s

#import <Foundation/Foundation.h>

#import "../test.h"

long global;
long array[2];

void callback(void *context, size_t i) {
  long n = global;
  array[i] = n + i;
  barrier_wait(&barrier);
}

int main(int argc, const char *argv[]) {
  barrier_init(&barrier, 2);
  fprintf(stderr, "start\n");
  dispatch_queue_t q = dispatch_queue_create("my.queue", DISPATCH_QUEUE_CONCURRENT);

  global = 42;

  dispatch_apply(100, q, ^(size_t i) {
    long n = global;
    array[i] = n + i;
    barrier_wait(&barrier);
  });

  for (int i = 0; i < 100; i++) {
    fprintf(stderr, "array[%d] = %ld\n", i, array[i]);
  }

  global = 43;

  dispatch_apply_f(100, q, NULL, &callback);

  fprintf(stderr, "done\n");
  return 0;
}

// CHECK: start
// CHECK: done
// CHECK-NOT: WARNING: ThreadSanitizer
