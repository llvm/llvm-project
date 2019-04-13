// RUN: %clang_tsan %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s --implicit-check-not='ThreadSanitizer'

#include <dispatch/dispatch.h>

#include <stdio.h>
#include <string.h>

long global = 42;

int main(int argc, const char *argv[]) {
  fprintf(stderr, "Hello world.\n");

  dispatch_queue_t q = dispatch_queue_create("my.queue", DISPATCH_QUEUE_SERIAL);
  dispatch_semaphore_t sem = dispatch_semaphore_create(0);

  const char *buffer = "buffer";
  size_t size = strlen(buffer);

  dispatch_data_t data = dispatch_data_create(buffer, size, q, ^{
    fprintf(stderr, "Data destructor.\n");
    global++;

    dispatch_semaphore_signal(sem);
  });
  dispatch_release(data);

  dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);

  data = dispatch_data_create(buffer, size, q, DISPATCH_DATA_DESTRUCTOR_DEFAULT);
  dispatch_release(data);

  fprintf(stderr, "Done.\n");
}

// CHECK: Hello world.
// CHECK: Data destructor.
// CHECK: Done.
