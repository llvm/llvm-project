// RUN: %clang_tsan %s -o %t -framework Foundation
// RUN: %env_tsan_opts=ignore_interceptors_accesses=1 %deflake %run %t 2>&1 | FileCheck %s

#import <Foundation/Foundation.h>

#import "../test.h"

long global;

int main(int argc, const char *argv[]) {
  barrier_init(&barrier, 2);
  fprintf(stderr, "start\n");
  dispatch_queue_t q = dispatch_queue_create("my.queue", DISPATCH_QUEUE_CONCURRENT);
  dispatch_apply(2, q, ^(size_t i) {
    global = i;
    barrier_wait(&barrier);
  });

  fprintf(stderr, "done\n");
  return 0;
}

// CHECK: start
// CHECK: WARNING: ThreadSanitizer: data race
// CHECK: Location is global 'global'
// CHECK: done
