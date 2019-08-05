// Checks the ASan memory address type debugging API, makes sure it returns
// the correct memory type for heap, stack, global and shadow addresses and
// that it correctly finds out which region (and name and size) the address
// belongs to.
// RUN: %clangxx_asan -O0 %s -o %t && %run %t 2>&1

#include <assert.h>
#include <sanitizer/asan_interface.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int global_var;

int main() {
  int local_var;
  char *heap_ptr = (char *)malloc(10);

  char name[100];
  void *region_address;
  size_t region_size;
  const char *type;

  type = __asan_locate_address(&global_var, name, 100,
                               &region_address, &region_size);
  assert(0 == strcmp(name, "global_var"));
  assert(0 == strcmp(type, "global"));
  assert(region_address == &global_var);
  assert(region_size == sizeof(global_var));

  type = __asan_locate_address((char *)(&global_var)+1, name, 100,
                               &region_address, &region_size);
  assert(0 == strcmp(name, "global_var"));
  assert(0 == strcmp(type, "global"));
  assert(region_address == &global_var);
  assert(region_size == sizeof(global_var));

  type = __asan_locate_address(&local_var, name, 100,
                               &region_address, &region_size);
  assert(0 == strcmp(name, "local_var"));
  assert(0 == strcmp(type, "stack"));
  assert(region_address == &local_var);
  assert(region_size == sizeof(local_var));

  type = __asan_locate_address((char *)(&local_var)+1, name, 100,
                               &region_address, &region_size);
  assert(0 == strcmp(name, "local_var"));
  assert(0 == strcmp(type, "stack"));
  assert(region_address == &local_var);
  assert(region_size == sizeof(local_var));

  type = __asan_locate_address(heap_ptr, name, 100,
                               &region_address, &region_size);
  assert(0 == strcmp(type, "heap"));
  assert(region_address == heap_ptr);
  assert(10 == region_size);

  type = __asan_locate_address(heap_ptr+1, name, 100,
                               &region_address, &region_size);
  assert(0 == strcmp(type, "heap"));
  assert(region_address == heap_ptr);
  assert(10 == region_size);

  size_t shadow_scale;
  size_t shadow_offset;
  __asan_get_shadow_mapping(&shadow_scale, &shadow_offset);

  uintptr_t shadow_ptr = (((uintptr_t)heap_ptr) >> shadow_scale)
                         + shadow_offset;
  type = __asan_locate_address((void *)shadow_ptr, NULL, 0, NULL, NULL);
  assert((0 == strcmp(type, "high shadow")) || 0 == strcmp(type, "low shadow"));

  uintptr_t shadow_gap = (shadow_ptr >> shadow_scale) + shadow_offset;
  type = __asan_locate_address((void *)shadow_gap, NULL, 0, NULL, NULL);
  assert(0 == strcmp(type, "shadow gap"));

  free(heap_ptr);

  return 0;
}
