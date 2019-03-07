// Check that calling dispatch_once from a report callback works.

// RUN: %clang_tsan %s -o %t
// RUN: not %run %t 2>&1 | FileCheck %s

// REQUIRES: dispatch

#include <dispatch/dispatch.h>

#include <pthread.h>
#include <stdio.h>

long g = 0;
long h = 0;
void f() {
  static dispatch_once_t onceToken;
  dispatch_once(&onceToken, ^{
    g++;
  });
  h++;
}

void __tsan_on_report() {
  fprintf(stderr, "Report.\n");
  f();
}

int main() {
  fprintf(stderr, "Hello world.\n");

  f();

  pthread_mutex_t mutex = {0};
  pthread_mutex_lock(&mutex);

  fprintf(stderr, "g = %ld.\n", g);
  fprintf(stderr, "h = %ld.\n", h);
  fprintf(stderr, "Done.\n");
}

// CHECK: Hello world.
// CHECK: Report.
// CHECK: g = 1
// CHECK: h = 2
// CHECK: Done.
