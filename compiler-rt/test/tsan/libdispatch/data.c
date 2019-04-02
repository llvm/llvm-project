// RUN: %clang_tsan %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s --implicit-check-not='ThreadSanitizer'

// TODO(yln): fails on one of our bots, need to investigate
// REQUIRES: disabled

#include <dispatch/dispatch.h>

#include <stdio.h>

long global;

int main(int argc, const char *argv[]) {
  fprintf(stderr, "Hello world.\n");

  dispatch_queue_t q = dispatch_queue_create("my.queue", DISPATCH_QUEUE_SERIAL);
  dispatch_semaphore_t sem = dispatch_semaphore_create(0);

  global = 44;
  dispatch_data_t data = dispatch_data_create("buffer", 6, q, ^{
    fprintf(stderr, "Data destructor.\n");
    global++;

    dispatch_semaphore_signal(sem);
  });
  dispatch_release(data);
  data = NULL;

  dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);

  data = dispatch_data_create("buffer", 6, q, DISPATCH_DATA_DESTRUCTOR_DEFAULT);
  dispatch_release(data);
  data = NULL;

  fprintf(stderr, "Done.\n");
}

// CHECK: Hello world.
// CHECK: Data destructor.
// CHECK: Done.
