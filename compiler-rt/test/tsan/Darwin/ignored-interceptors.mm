// Check that ignore_interceptors_accesses=1 supresses reporting races from
// system libraries on OS X. There are currently false positives coming from
// libxpc, libdispatch, CoreFoundation and others, because these libraries use
// TSan-invisible atomics as synchronization.

// RUN: %clang_tsan %s -o %t -framework Foundation

// Check that without the flag, there are false positives.
// RUN: %deflake %run %t 2>&1 | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-RACE

// With ignore_interceptors_accesses=1, no races are reported.
// RUN: %env_tsan_opts=ignore_interceptors_accesses=1 %run %t 2>&1 | FileCheck %s

// With ignore_interceptors_accesses=1, races in user's code are still reported.
// RUN: %env_tsan_opts=ignore_interceptors_accesses=1 %deflake %run %t race 2>&1 | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-RACE

#import <Foundation/Foundation.h>

#import "../test.h"

long global;

void *Thread1(void *x) {
  barrier_wait(&barrier);
  global = 42;
  return NULL;
}

void *Thread2(void *x) {
  global = 43;
  barrier_wait(&barrier);
  return NULL;
}

int main(int argc, char *argv[]) {
  fprintf(stderr, "Hello world.\n");
  
  // NSUserDefaults uses XPC which triggers the false positive.
  NSDictionary *d = [[NSUserDefaults standardUserDefaults] dictionaryRepresentation];

  if (argc > 1 && strcmp(argv[1], "race") == 0) {
    barrier_init(&barrier, 2);
    pthread_t t[2];
    pthread_create(&t[0], NULL, Thread1, NULL);
    pthread_create(&t[1], NULL, Thread2, NULL);
    pthread_join(t[0], NULL);
    pthread_join(t[1], NULL);
  }

  fprintf(stderr, "Done.\n");
}

// CHECK: Hello world.
// CHECK-RACE: SUMMARY: ThreadSanitizer: data race
// CHECK: Done.
