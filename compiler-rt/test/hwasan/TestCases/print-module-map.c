// RUN: %clang_hwasan %s -o %t && %env_hwasan_opts=print_module_map=1 %run %t 2>&1 | FileCheck %s --check-prefixes=EXIT,NOMORE
// RUN: %clang_hwasan %s -DBUG -o %t && %env_hwasan_opts=print_module_map=1 not %run %t 2>&1 | FileCheck %s --check-prefixes=EXIT,NOMORE
// RUN: %clang_hwasan %s -DBUG -fsanitize-recover=hwaddress -o %t && %env_hwasan_opts=print_module_map=1,halt_on_error=0 not %run %t 2>&1 | FileCheck %s --check-prefixes=EXIT,NOMORE
// RUN: %clang_hwasan %s -DBUG -fsanitize-recover=hwaddress -o %t && %env_hwasan_opts=print_module_map=2,halt_on_error=0 not %run %t 2>&1 | FileCheck %s --check-prefixes=BUG1,BUG2,EXIT,NOMORE

#include <stdlib.h>
#include <stdio.h>
#include <sanitizer/hwasan_interface.h>

int main() {
  __hwasan_enable_allocator_tagging();
#ifdef BUG
  char * volatile x = (char*)malloc(40);
  free(x);
  free(x);
  free(x);
#endif
  __hwasan_disable_allocator_tagging();
  // BUG1: Process memory map follows:
  // BUG1: print-module-map
  // BUG1: End of process memory map.

  // BUG2: Process memory map follows:
  // BUG2: print-module-map
  // BUG2: End of process memory map.

  // EXIT: Process memory map follows:
  // EXIT: print-module-map
  // EXIT: End of process memory map.

  // NOMORE-NOT: Process memory map follows:
}
