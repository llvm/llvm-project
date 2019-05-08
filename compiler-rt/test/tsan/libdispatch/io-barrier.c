// RUN: %clang_tsan %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s

#include <dispatch/dispatch.h>

#include <stdio.h>
#include <stdlib.h>

dispatch_queue_t queue;
dispatch_data_t data;
dispatch_semaphore_t sem;
const char *path;

long my_global = 0;

int main(int argc, const char *argv[]) {
  fprintf(stderr, "Hello world.\n");
  
  queue = dispatch_queue_create("my.queue", DISPATCH_QUEUE_CONCURRENT);
  sem = dispatch_semaphore_create(0);
  path = tempnam(NULL, "libdispatch-io-barrier");
  char buf[1000];
  data = dispatch_data_create(buf, sizeof(buf), NULL, DISPATCH_DATA_DESTRUCTOR_DEFAULT);
  
  dispatch_io_t channel = dispatch_io_create_with_path(DISPATCH_IO_STREAM, path, O_CREAT | O_WRONLY, 0666, queue, ^(int error) { });
  if (! channel) abort();
  dispatch_io_set_high_water(channel, 1);

  for (int i = 0; i < 1000; i++) {
    dispatch_io_barrier(channel, ^{
      my_global = 42;
    });
  }

  dispatch_io_barrier(channel, ^{
    my_global = 43;

    dispatch_semaphore_signal(sem);
  });

  dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);
  dispatch_io_close(channel, 0);
  
  fprintf(stderr, "Done.\n");
  return 0;
}

// CHECK: Hello world.
// CHECK-NOT: WARNING: ThreadSanitizer
// CHECK: Done.
