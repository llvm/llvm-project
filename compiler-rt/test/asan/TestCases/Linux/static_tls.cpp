// REQUIRES: asan-64-bits
// Regression test: __tls_get_addr interceptor must recognize static TLS.
//
// RUN: %clangxx_asan -DSHARED %s -shared -o %t-so.so -fPIC
// RUN: %clangxx_asan %s -ldl -pthread -o %t %t-so.so
// RUN: %env_asan_opts=verbosity=2 %run %t 2>&1 | FileCheck %s

// CHECK: before
// CHECK: __tls_get_addr: static tls
// CHECK: after

// XFAIL: aarch64
// binutils 2.26 has a change that causes this test to fail on powerpc64.
// UNSUPPORTED: powerpc64 

#ifndef SHARED
#include <stdio.h>

unsigned *f();
int main(int argc, char *argv[]) {
  fprintf(stderr, "before\n");
  f();
  fprintf(stderr, "after\n");
  return 0;
}
#else  // SHARED
static __thread unsigned ThreadLocal;
unsigned *f() {
  return &ThreadLocal;
}
#endif
