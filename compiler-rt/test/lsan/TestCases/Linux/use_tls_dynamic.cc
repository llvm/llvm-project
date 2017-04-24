// Test that dynamically allocated TLS space is included in the root set.
// RUN: LSAN_BASE="report_objects=1:use_stacks=0:use_registers=0:use_ld_allocations=0"
// RUN: %clangxx %s -DBUILD_DSO -fPIC -shared -o %t-so.so
// RUN: %clangxx_lsan %s -o %t
// RUN: %env_lsan_opts=$LSAN_BASE:"use_tls=0" not %run %t 2>&1 | FileCheck %s
// RUN: %env_lsan_opts=$LSAN_BASE:"use_tls=1" %run %t 2>&1
// RUN: %env_lsan_opts="" %run %t 2>&1
// UNSUPPORTED: i386-linux,i686-linux,arm

#ifndef BUILD_DSO
#include <assert.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include "sanitizer_common/print_address.h"

int main(int argc, char *argv[]) {
  std::string path = std::string(argv[0]) + "-so.so";

  void *handle = dlopen(path.c_str(), RTLD_LAZY);
  assert(handle != 0);
  typedef void **(* store_t)(void *p);
  store_t StoreToTLS = (store_t)dlsym(handle, "StoreToTLS");
  assert(dlerror() == 0);

  void *p = malloc(1337);
  // If we don't  know about dynamic TLS, we will return a false leak above.
  void **p_in_tls = StoreToTLS(p);
  assert(*p_in_tls == p);
  print_address("Test alloc: ", 1, p);
  return 0;
}
// CHECK: Test alloc: [[ADDR:0x[0-9,a-f]+]]
// CHECK: LeakSanitizer: detected memory leaks
// CHECK: [[ADDR]] (1337 bytes)
// CHECK: SUMMARY: {{(Leak|Address)}}Sanitizer:

#else  // BUILD_DSO
// A loadable module with a large thread local section, which would require
// allocation of a new TLS storage chunk when loaded with dlopen(). We use it
// to test the reachability of such chunks in LSan tests.

// This must be large enough that it doesn't fit into preallocated static TLS
// space (see STATIC_TLS_SURPLUS in glibc).
__thread void *huge_thread_local_array[(1 << 20) / sizeof(void *)]; // NOLINT

extern "C" void **StoreToTLS(void *p) {
  huge_thread_local_array[0] = p;
  return &huge_thread_local_array[0];
}
#endif  // BUILD_DSO
