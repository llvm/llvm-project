// RUN: %clang_tsan %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s --implicit-check-not='ThreadSanitizer'

#include <dispatch/dispatch.h>

#include <stdio.h>
#include <stdlib.h>

long my_global = 0;

int main(int argc, const char *argv[]) {
  fprintf(stderr, "Hello world.\n");
  
  dispatch_queue_t queue = dispatch_queue_create("my.queue", DISPATCH_QUEUE_CONCURRENT);
  dispatch_semaphore_t sem = dispatch_semaphore_create(0);
  const char *path = tempnam(NULL, "libdispatch-io-cleanup-");
  dispatch_io_t channel;
  
  dispatch_fd_t fd = open(path, O_CREAT | O_WRONLY, 0666);
  my_global++;
  channel = dispatch_io_create(DISPATCH_IO_STREAM, fd, queue, ^(int error) {
    my_global++;
    dispatch_semaphore_signal(sem);
  });
  if (! channel) abort();
  my_global++;
  dispatch_io_close(channel, 0);
  dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);
  
  my_global++;
  channel = dispatch_io_create_with_path(DISPATCH_IO_STREAM, path, O_CREAT | O_WRONLY, 0666, queue, ^(int error) {
    my_global++;
    dispatch_semaphore_signal(sem);
  });
  if (! channel) abort();
  my_global++;
  dispatch_io_close(channel, 0);
  dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);
  
  my_global++;
  dispatch_io_t other_channel = dispatch_io_create_with_path(DISPATCH_IO_STREAM, path, O_CREAT | O_WRONLY, 0666, queue, ^(int error) { });
  channel = dispatch_io_create_with_io(DISPATCH_IO_STREAM, other_channel, queue, ^(int error) {
    my_global++;
    dispatch_semaphore_signal(sem);
  });
  if (! channel) abort();
  my_global++;
  dispatch_io_close(channel, 0);
  dispatch_io_close(other_channel, 0);
  dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);
  
  fprintf(stderr, "Done.\n");
  return 0;
}

// CHECK: Hello world.
// CHECK: Done.
