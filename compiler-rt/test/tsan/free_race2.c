// RUN: %clang_tsan -O1 %s -o %t && %deflake %run %t | FileCheck %s
// RUN: %clang_tsan -O1 -DACCESS_OFFSET=4 %s -o %t && %deflake %run %t | FileCheck %s
#include <stdlib.h>

#ifndef ACCESS_OFFSET
#define ACCESS_OFFSET 0
#endif

__attribute__((noinline)) void foo(void *mem) {
  free(mem);
}

__attribute__((noinline)) void baz(void *mem) {
  free(mem);
}

__attribute__((noinline)) void bar(void *mem) {
  *(long*)((char*)mem + ACCESS_OFFSET) = 42;
}

int main() {
  void *mem = malloc(100);
  baz(mem);
  mem = malloc(100);
  foo(mem);
  bar(mem);
  return 0;
}

// CHECK: WARNING: ThreadSanitizer: heap-use-after-free
// CHECK:   Write of size {{.*}} at {{.*}} by main thread:
// CHECK:     #0 bar
// CHECK:     #1 main
// CHECK:   Previous write of size 8 at {{.*}} by main thread:
// CHECK:     #0 free
// CHECK:     #{{1|2}} foo
// CHECK:     #{{2|3}} main
