// RUN: %clang_hwasan -O0 %s -o %t
// RUN: %env_hwasan_opts=malloc_bisect_left=0,malloc_bisect_right=0          not %run %t 2>&1 | \
// RUN:     FileCheck %s --check-prefix=CRASH
// RUN: %env_hwasan_opts=malloc_bisect_left=1000,malloc_bisect_right=999         %run %t 2>&1
// RUN: %env_hwasan_opts=malloc_bisect_left=0,malloc_bisect_right=4294967295 not %run %t 2>&1 | \
// RUN:     FileCheck %s --check-prefix=CRASH
// RUN: %env_hwasan_opts=malloc_bisect_left=0,malloc_bisect_right=4294967295,malloc_bisect_dump=1 not %run %t 2>&1 | \
// RUN:     FileCheck %s --check-prefixes=CRASH,DUMP

// FIXME: Recursive call into malloc from stack->Print(). Or maybe this feature
// needs to be removed. We don't use it, but it's in the hot code.
// UNSUPPORTED: internal_symbolizer

#include <stdlib.h>
#include <stdio.h>
#include <sanitizer/hwasan_interface.h>

int main() {
  __hwasan_enable_allocator_tagging();
  // DUMP: [alloc] {{.*}} 10{{$}}
  // DUMP: in main{{.*}}malloc_bisect.c
  char * volatile p = (char*)malloc(10);
  // CRASH: HWAddressSanitizer: tag-mismatch on address
  // CRASH: in main{{.*}}malloc_bisect.c
  char volatile x = p[16];
  free(p);
  __hwasan_disable_allocator_tagging();

  return 0;
}
