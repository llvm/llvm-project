// RUN: %clang_tsan %s -o %t -framework Foundation
// RUN: %deflake %run %t 2>&1 | FileCheck %s

#import <Foundation/Foundation.h>

#import "../test.h"

dispatch_queue_t queue;
dispatch_data_t data;
dispatch_semaphore_t sem;
const char *path;

long my_global = 0;

int main(int argc, const char *argv[]) {
  fprintf(stderr, "Hello world.\n");
  print_address("addr=", 1, &my_global);
  barrier_init(&barrier, 2);

  queue = dispatch_queue_create("my.queue", DISPATCH_QUEUE_CONCURRENT);
  sem = dispatch_semaphore_create(0);
  path = tempnam(NULL, "libdispatch-io-barrier-race-");
  char buf[1000];
  data = dispatch_data_create(buf, sizeof(buf), NULL, DISPATCH_DATA_DESTRUCTOR_DEFAULT);
  
  dispatch_io_t channel = dispatch_io_create_with_path(DISPATCH_IO_STREAM, path, O_CREAT | O_WRONLY, 0666, queue, ^(int error) { });
  if (! channel) abort();
  dispatch_io_set_high_water(channel, 1);

  dispatch_io_write(channel, 0, data, queue, ^(bool done, dispatch_data_t remainingData, int error) {
    if (error) abort();
    my_global = 42;
    barrier_wait(&barrier);
  });

  dispatch_io_barrier(channel, ^{
    barrier_wait(&barrier);
    my_global = 43;

    dispatch_semaphore_signal(sem);
  });

  dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);
  dispatch_io_close(channel, 0);
  
  fprintf(stderr, "Done.\n");
  return 0;
}

// CHECK: Hello world.
// CHECK: addr=[[ADDR:0x[0-9,a-f]+]]
// CHECK: WARNING: ThreadSanitizer: data race
// CHECK: Location is global 'my_global' {{(of size 8 )?}}at [[ADDR]] (gcd-io-barrier-race.mm.tmp+0x{{[0-9,a-f]+}})
// CHECK: Done.
