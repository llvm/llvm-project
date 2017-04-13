// Test that statically allocated thread-specific storage is included in the root set.
// RUN: LSAN_BASE="detect_leaks=1:report_objects=1:use_stacks=0:use_registers=0"
// RUN: %clangxx_lsan %s -o %t
// RUN: LSAN_OPTIONS=$LSAN_BASE:"use_tls=0" not %run %t 2>&1 | FileCheck %s
// RUN: LSAN_OPTIONS=$LSAN_BASE:"use_tls=1" %run %t 2>&1
// RUN: LSAN_OPTIONS="" %run %t 2>&1

#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include "sanitizer_common/print_address.h"

// From glibc: this many keys are stored in the thread descriptor directly.
const unsigned PTHREAD_KEY_2NDLEVEL_SIZE = 32;

int main() {
  pthread_key_t key;
  int res;
  res = pthread_key_create(&key, NULL);
  assert(res == 0);
  assert(key < PTHREAD_KEY_2NDLEVEL_SIZE);
  void *p = malloc(1337);
  res = pthread_setspecific(key, p);
  assert(res == 0);
  print_address("Test alloc: ", 1, p);
  return 0;
}
// CHECK: Test alloc: [[ADDR:0x[0-9,a-f]+]]
// CHECK: LeakSanitizer: detected memory leaks
// CHECK: [[ADDR]] (1337 bytes)
// CHECK: SUMMARY: {{(Leak|Address)}}Sanitizer:
