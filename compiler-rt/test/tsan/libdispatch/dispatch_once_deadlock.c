// Check that calling dispatch_once from a report callback works.

// RUN: %clang_tsan %s -o %t
// RUN: not %env_tsan_opts=ignore_noninstrumented_modules=0 %run %t 2>&1 | FileCheck %s

#include <dispatch/dispatch.h>

#include <pthread.h>
#include <stdio.h>

long g = 0;
long h = 0;

__attribute__((disable_sanitizer_instrumentation))
void f() {
  static dispatch_once_t onceToken;
  dispatch_once(&onceToken, ^{
    g++;
  });
  h++;
}

__attribute__((disable_sanitizer_instrumentation))
void __tsan_on_report() {
  fprintf(stderr, "Report.\n");
  f();
}

int main() {
  fprintf(stderr, "Hello world.\n");

  f();

  pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
  pthread_mutex_unlock(&mutex); // Unlock of an unlocked mutex

  fprintf(stderr, "g = %ld.\n", g);
  fprintf(stderr, "h = %ld.\n", h);
  fprintf(stderr, "Done.\n");
}

// CHECK: Hello world.
// CHECK: Report.
// CHECK: g = 1
// CHECK: h = 2
// CHECK: Done.
