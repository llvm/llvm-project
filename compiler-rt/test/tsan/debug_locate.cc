// RUN: %clangxx_tsan -O0 %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s

#include <stdio.h>
#include <stdlib.h>

extern "C" const char *
__tsan_locate_address(void *addr, char *name, size_t name_size,
                      void **region_address_ptr, size_t *region_size_ptr);

long global_var;

int main() {
  long stack_var;
  void *heap_var = malloc(10);

  fprintf(stderr, "stack_var = %p\n", &stack_var);
  fprintf(stderr, "global_var = %p\n", &global_var);
  fprintf(stderr, "heap_var = %p\n", heap_var);
  // CHECK: stack_var = [[STACK_VAR:0x[0-9a-f]+]]
  // CHECK: global_var = [[GLOBAL_VAR:0x[0-9a-f]+]]
  // CHECK: heap_var = [[HEAP_VAR:0x[0-9a-f]+]]

  const char *type;
  char name[128];
  void *start;
  size_t size;
  type = __tsan_locate_address(&stack_var, name, 128, &start, &size);
  fprintf(stderr, "type: %s\n", type);
  // CHECK: type: stack

  type = __tsan_locate_address(&global_var, name, 128, &start, &size);
  fprintf(stderr, "type: %s, name = %s, start = %p, size = %zu\n", type, name,
          start, size);
  // CHECK: type: global, name = global_var, start = [[GLOBAL_VAR]], size = {{8|0}}

  type = __tsan_locate_address(heap_var, name, 128, &start, &size);
  fprintf(stderr, "type: %s, start = %p, size = %zu\n", type, start, size);
  // CHECK: type: heap, start = [[HEAP_VAR]], size = 10

  free(heap_var);
  return 0;
}
