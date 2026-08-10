// RUN: %clang_tsan %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s --implicit-check-not='ThreadSanitizer'

#include <dispatch/dispatch.h>

#include <stdio.h>

long my_global = 0;

int main(int argc, const char *argv[]) {
  fprintf(stderr, "Hello world.\n");

  dispatch_queue_t queue = dispatch_queue_create("my.queue", DISPATCH_QUEUE_SERIAL);
  dispatch_semaphore_t sem = dispatch_semaphore_create(0);

  const char *path = tempnam(NULL, "libdispatch-fd-");

  dispatch_io_t channel = dispatch_io_create_with_path(DISPATCH_IO_STREAM, path, O_CREAT | O_WRONLY,
      0666, queue, ^(int error) { });
  dispatch_io_set_high_water(channel, 1);

  char buf[1000];
  dispatch_data_t data = dispatch_data_create(buf, sizeof(buf), NULL, DISPATCH_DATA_DESTRUCTOR_DEFAULT);

  my_global++;
  dispatch_io_write(channel, 0, data, queue, ^(bool done, dispatch_data_t remainingData, int error) {
    my_global++;
    dispatch_async(queue, ^{
      my_global++;
      if (done) {
        dispatch_semaphore_signal(sem);
      }
    });
  });

  dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);
  my_global++;
  dispatch_io_close(channel, 0);
  channel = dispatch_io_create_with_path(DISPATCH_IO_STREAM, path, O_RDONLY,
      0, queue, ^(int error) { });
  dispatch_io_set_high_water(channel, 1);

  my_global++;
  dispatch_io_read(channel, 0, SIZE_MAX, queue, ^(bool done, dispatch_data_t remainingData, int error) {
    my_global++;
    dispatch_async(queue, ^{
      my_global++;
      if (done) {
        dispatch_semaphore_signal(sem);
      }
    });
  });

  dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);
  my_global++;
  fprintf(stderr, "Done.\n");
  return 0;
}

// CHECK: Hello world.
// CHECK: Done.
