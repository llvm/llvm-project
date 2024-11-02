// Test that dynamically allocated thread-specific storage is included in the root set.
// RUN: %clangxx_lsan %s -o %t
// RUN: %env_lsan_opts="report_objects=1:use_stacks=0:use_registers=0:use_tls=0" not %run %t 2>&1 | FileCheck %s
// RUN: %env_lsan_opts="report_objects=1:use_stacks=0:use_registers=0:use_tls=1" %run %t 2>&1
// RUN: %env_lsan_opts="" %run %t 2>&1

// Investigate why it does not fail with use_tls=0
// UNSUPPORTED: arm-linux || armhf-linux

#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include "sanitizer_common/print_address.h"

// From glibc: this many keys are stored in the thread descriptor directly.
const unsigned PTHREAD_KEY_2NDLEVEL_SIZE = 32;

int main() {
  static const unsigned kDummyKeysCount = PTHREAD_KEY_2NDLEVEL_SIZE;
  int res;
  pthread_key_t dummy_keys[kDummyKeysCount];
  for (unsigned i = 0; i < kDummyKeysCount; i++) {
    res = pthread_key_create(&dummy_keys[i], NULL);
    assert(res == 0);
  }
  pthread_key_t key;
  res = pthread_key_create(&key, NULL);
  assert(key >= PTHREAD_KEY_2NDLEVEL_SIZE);
  assert(res == 0);
  void *p  = malloc(1337);
  res = pthread_setspecific(key, p);
  assert(res == 0);
  print_address("Test alloc: ", 1, p);
  return 0;
}
// CHECK: Test alloc: [[ADDR:0x[0-9,a-f]+]]
// CHECK: LeakSanitizer: detected memory leaks
// CHECK: [[ADDR]] (1337 bytes)
// CHECK: SUMMARY: {{(Leak|Address)}}Sanitizer:
