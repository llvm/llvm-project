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

void test_dispatch_io_write() {
  dispatch_io_t channel = dispatch_io_create_with_path(DISPATCH_IO_STREAM, path, O_CREAT | O_WRONLY, 0666, queue, ^(int error) { });
  if (! channel) abort();
  dispatch_io_set_high_water(channel, 1);
  
  my_global++;
  dispatch_io_write(channel, 0, data, queue, ^(bool done, dispatch_data_t remainingData, int error) {
    if (error) abort();
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
}

void test_dispatch_write() {
  dispatch_fd_t fd = open(path, O_CREAT | O_WRONLY, 0666);
  if (fd == -1) abort();
  
  my_global++;
  dispatch_write(fd, data, queue, ^(dispatch_data_t data, int error) {
    if (error) abort();
    my_global++;
    dispatch_async(queue, ^{
      my_global++;
      
      dispatch_semaphore_signal(sem);
    });
  });
  
  dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);
  my_global++;
  close(fd);
}

void test_dispatch_io_read() {
  dispatch_io_t channel = dispatch_io_create_with_path(DISPATCH_IO_STREAM, path, O_RDONLY,
                       0, queue, ^(int error) { });
  dispatch_io_set_high_water(channel, 1);
  
  my_global++;
  dispatch_io_read(channel, 0, SIZE_MAX, queue, ^(bool done, dispatch_data_t remainingData, int error) {
    if (error) abort();
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
}

void test_dispatch_read() {
  dispatch_fd_t fd = open(path, O_RDONLY, 0);
  if (fd == -1) abort();
  
  my_global++;
  dispatch_read(fd, SIZE_MAX, queue, ^(dispatch_data_t data, int error) {
    if (error) abort();
    my_global++;
    dispatch_async(queue, ^{
      my_global++;
      dispatch_semaphore_signal(sem);
    });
  });
  
  dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);
  my_global++;
  close(fd);
}

int main(int argc, const char *argv[]) {
  fprintf(stderr, "Hello world.\n");
  
  queue = dispatch_queue_create("my.queue", DISPATCH_QUEUE_SERIAL);
  sem = dispatch_semaphore_create(0);
  path = tempnam(NULL, "libdispatch-io-");
  char buf[1000];
  data = dispatch_data_create(buf, sizeof(buf), NULL, DISPATCH_DATA_DESTRUCTOR_DEFAULT);
  
  test_dispatch_io_write();
  test_dispatch_write();
  test_dispatch_io_read();
  test_dispatch_read();
  
  fprintf(stderr, "Done.\n");
  return 0;
}

// CHECK: Hello world.
// CHECK-NOT: WARNING: ThreadSanitizer
// CHECK: Done.
