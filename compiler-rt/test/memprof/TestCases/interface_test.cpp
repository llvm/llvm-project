// Check that user may include MemProf interface header.
// Also check that interfaces declared in the sanitizer's allocator_interface
// are defined for MemProf.
// RUN: %clangxx_memprof %s -o %t -DMEMPROF && %run %t
// RUN: %clangxx_memprof -x c %s -o %t -DMEMPROF && %run %t
// RUN: %clang %s -pie -o %t && %run %t
// RUN: %clang -x c %s -pie -o %t && %run %t
#include <sanitizer/allocator_interface.h>
#include <sanitizer/memprof_interface.h>
#include <stdlib.h>

int main() {
  int *p = (int *)malloc(10 * sizeof(int));
#ifdef MEMPROF
  __sanitizer_get_estimated_allocated_size(8);
  __sanitizer_get_ownership(p);
  __sanitizer_get_allocated_size(p);
  __sanitizer_get_current_allocated_bytes();
  __sanitizer_get_heap_size();
  __sanitizer_get_free_bytes();
  __sanitizer_get_unmapped_bytes();
  // malloc and free hooks are tested by the malloc_hook.cpp test.
#endif
  return 0;
}
