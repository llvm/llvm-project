// Check that we don't crash when dispatch_main calls pthread_exit which
// quits the main thread.

// RUN: %clang_tsan %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s

#include <dispatch/dispatch.h>

#include <stdio.h>
#include <stdlib.h>

int main() {
  fprintf(stderr,"Hello world");

  dispatch_queue_t q = dispatch_queue_create("my.queue", DISPATCH_QUEUE_SERIAL);

  dispatch_async(q, ^{
    fprintf(stderr,"1");
  });

  dispatch_async(q, ^{
    fprintf(stderr,"2");
  });

  dispatch_async(q, ^{
    fprintf(stderr,"3");

    dispatch_async(dispatch_get_main_queue(), ^{
      fprintf(stderr,"Done.");
      sleep(1);
      exit(0);
    });
  });

  dispatch_main();
}

// CHECK: Hello world
// CHECK: Done.
// CHECK-NOT: WARNING: ThreadSanitizer
// CHECK-NOT: CHECK failed
