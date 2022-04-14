// RUN: %clangxx -O0 -g %s -o %t && %run %t

// Test requires platform with thread local support with no dependency on malloc.
// UNSUPPORTED: android
// UNSUPPORTED: ios
// UNSUPPORTED: darwin

#include <assert.h>
#include <sanitizer/allocator_interface.h>
#include <sanitizer/coverage_interface.h>
#include <stdlib.h>

static int hooks;
extern "C" {

void __sanitizer_malloc_hook(const volatile void *ptr, size_t sz) { ++hooks; }

void __sanitizer_free_hook(const volatile void *ptr) { ++hooks; }

} // extern "C"

void MallocHook(const volatile void *ptr, size_t sz) { ++hooks; }
void FreeHook(const volatile void *ptr) { ++hooks; }

int main() {
  int before;

  before = hooks;
  __sanitizer_print_stack_trace();
  assert(before == hooks);

  __sanitizer_install_malloc_and_free_hooks(MallocHook, FreeHook);
  before = hooks;
  __sanitizer_print_stack_trace();
  assert(before == hooks);

  return 0;
}
