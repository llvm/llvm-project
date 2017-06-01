// Check that ignore_noninstrumented_modules=1 supresses races from system libraries on OS X.

// RUN: %clang_tsan %s -o %t -framework Foundation

// Check that without the flag, there are false positives.
// RUN: %env_tsan_opts=ignore_noninstrumented_modules=0 %deflake %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-RACE

// With ignore_noninstrumented_modules=1, no races are reported.
// RUN: %env_tsan_opts=ignore_noninstrumented_modules=1 %run %t 2>&1 | FileCheck %s

// With ignore_noninstrumented_modules=1, races in user's code are still reported.
// RUN: %env_tsan_opts=ignore_noninstrumented_modules=1 %deflake %run %t race 2>&1 | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-RACE

#import <Foundation/Foundation.h>

#import "../test.h"

char global_buf[64];

void *Thread1(void *x) {
  barrier_wait(&barrier);
  strcpy(global_buf, "hello world");
  return NULL;
}

void *Thread2(void *x) {
  strcpy(global_buf, "world hello");
  barrier_wait(&barrier);
  return NULL;
}

int main(int argc, char *argv[]) {
  fprintf(stderr, "Hello world.\n");
  
  // NSUserDefaults uses XPC which triggers the false positive.
  NSDictionary *d = [[NSUserDefaults standardUserDefaults] dictionaryRepresentation];
  fprintf(stderr, "d = %p\n", d);

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
