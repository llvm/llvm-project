// RUN: %clangxx -O2 %s -o %t && %run %t 2>&1

// Malloc/free hooks are not supported on Windows.
// XFAIL: target={{.*windows-msvc.*}}

// Must not be implemented, no other reason to install interceptors.
// XFAIL: ubsan

#include <assert.h>
#include <sanitizer/allocator_interface.h>
#include <stdlib.h>
#include <unistd.h>

extern "C" {
const volatile void *global_ptr;

// Note: avoid calling functions that allocate memory in malloc/free
// to avoid infinite recursion.
void __sanitizer_malloc_hook(const volatile void *ptr, size_t sz) {
  if (__sanitizer_get_ownership(ptr) && sz == sizeof(int)) {
    global_ptr = ptr;
    assert(__sanitizer_get_allocated_size_fast(ptr) == sizeof(int));
  }
}
void __sanitizer_free_hook(const volatile void *ptr) {
  if (__sanitizer_get_ownership(ptr) && ptr == global_ptr)
    assert(__sanitizer_get_allocated_size_fast(ptr) == sizeof(int));
}
} // extern "C"

volatile int *x;

// Call this function with uninitialized arguments to poison
// TLS shadow for function parameters before calling operator
// new and, eventually, user-provided hook.
__attribute__((noinline)) void allocate(int *unused1, int *unused2) {
  x = reinterpret_cast<int *>(malloc(sizeof(int)));
}

int main() {
  int *undef1, *undef2;
  allocate(undef1, undef2);

  // Check that malloc hook was called with correct argument.
  if (global_ptr != (void *)x) {
    _exit(1);
  }

  *x = -8;
  free((void *)x);

  return 0;
}
