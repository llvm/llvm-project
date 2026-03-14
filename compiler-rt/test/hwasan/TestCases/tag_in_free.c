// RUN: %clang_hwasan -O0 %s -DMALLOC -DFREE -o %t.mf
// RUN: %env_hwasan_opts=tag_in_malloc=0,tag_in_free=1 not %run %t.mf 2>&1 | FileCheck %s --check-prefixes=FREE
// RUN: %env_hwasan_opts=tag_in_malloc=1,tag_in_free=1 not %run %t.mf 2>&1 | FileCheck %s --check-prefixes=MALLOC
// RUN: %env_hwasan_opts=tag_in_malloc=1,tag_in_free=0 not %run %t.mf 2>&1 | FileCheck %s --check-prefixes=MALLOC
// RUN: %env_hwasan_opts=tag_in_malloc=0,tag_in_free=0     %run %t.mf 2>&1

// RUN: %clang_hwasan -O0 %s -DFREE -o %t.f
// RUN: %env_hwasan_opts=tag_in_malloc=0,tag_in_free=1 not %run %t.f 2>&1 | FileCheck %s --check-prefixes=FREE
// RUN: %env_hwasan_opts=tag_in_malloc=1,tag_in_free=1 not %run %t.f 2>&1 | FileCheck %s --check-prefixes=FREE
// RUN: %env_hwasan_opts=tag_in_malloc=1,tag_in_free=0     %run %t.f 2>&1
// RUN: %env_hwasan_opts=tag_in_malloc=0,tag_in_free=0     %run %t.f 2>&1

// RUN: %clang_hwasan -O0 %s -DMALLOC -o %t.m
// RUN: %env_hwasan_opts=tag_in_malloc=0,tag_in_free=1     %run %t.m 2>&1
// RUN: %env_hwasan_opts=tag_in_malloc=1,tag_in_free=1 not %run %t.m 2>&1 | FileCheck %s --check-prefixes=MALLOC
// RUN: %env_hwasan_opts=tag_in_malloc=1,tag_in_free=0 not %run %t.m 2>&1 | FileCheck %s --check-prefixes=MALLOC
// RUN: %env_hwasan_opts=tag_in_malloc=0,tag_in_free=0     %run %t.m 2>&1

#include <stdlib.h>
#include <stdio.h>
#include <sanitizer/hwasan_interface.h>

int main() {
  __hwasan_enable_allocator_tagging();
  // Loop for a while to make sure that the memory for the test below is reused after an earlier free(),
  // and is potentially tagged (when tag_in_free == 1).
  for (int i = 0; i < 100; ++i) {
    char * volatile p = (char*)malloc(10);
    free(p);
  }

  char * volatile p = (char*)malloc(10);
#ifdef MALLOC
  // MALLOC: READ of size 1 at
  // MALLOC: is located 6 bytes after a 10-byte region
  // MALLOC: allocated by thread T0 here:
  char volatile x = p[16];
#endif
  free(p);
#ifdef FREE
  // FREE: READ of size 1 at
  // FREE: is located 0 bytes inside a 10-byte region
  // FREE: freed by thread T0 here:
  // FREE: previously allocated by thread T0 here:
  char volatile y = p[0];
#endif

  __hwasan_disable_allocator_tagging();

  return 0;
}
