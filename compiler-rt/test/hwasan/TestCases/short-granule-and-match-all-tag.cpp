// RUN: %clang_hwasan -mllvm -hwasan-match-all-tag=0 %s -o %t && %run %t

#include <sanitizer/hwasan_interface.h>
#include <stdlib.h>

int main() {
  __hwasan_enable_allocator_tagging();
  char *x = (char *)malloc(40);
  char volatile z = *x;
  free(x);
  return 0;
}
