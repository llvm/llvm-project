// RUN: %clangxx_nsan %s -o %t && %run %t 2>&1 | FileCheck %s

#include <sanitizer/allocator_interface.h>
#include <stdlib.h>
#include <unistd.h>

extern "C" {
const volatile void *global_ptr;

#define WRITE(s) write(1, s, sizeof(s))

// Note: avoid calling functions that allocate memory in malloc/free
// to avoid infinite recursion.
void __sanitizer_malloc_hook(const volatile void *ptr, size_t sz) {
  if (__sanitizer_get_ownership(ptr) && sz == 4) {
    WRITE("MallocHook\n");
    global_ptr = ptr;
  }
}
void __sanitizer_free_hook(const volatile void *ptr) {
  if (__sanitizer_get_ownership(ptr) && ptr == global_ptr)
    WRITE("FreeHook\n");
}
} // extern "C"

volatile int *x;

void MallocHook1(const volatile void *ptr, size_t sz) { WRITE("MH1\n"); }
void MallocHook2(const volatile void *ptr, size_t sz) { WRITE("MH2\n"); }
void FreeHook1(const volatile void *ptr) { WRITE("FH1\n"); }
void FreeHook2(const volatile void *ptr) { WRITE("FH2\n"); }
// Call this function with uninitialized arguments to poison
// TLS shadow for function parameters before calling operator
// new and, eventually, user-provided hook.
__attribute__((noinline)) void allocate(int *unused1, int *unused2) {
  x = new int;
}

int main() {
  __sanitizer_install_malloc_and_free_hooks(MallocHook1, FreeHook1);
  __sanitizer_install_malloc_and_free_hooks(MallocHook2, FreeHook2);
  int *undef1, *undef2;
  allocate(undef1, undef2);
  // CHECK: MallocHook
  // CHECK: MH1
  // CHECK: MH2
  // Check that malloc hook was called with correct argument.
  if (global_ptr != (void *)x) {
    _exit(1);
  }
  *x = 0;
  delete x;
  // CHECK: FreeHook
  // CHECK: FH1
  // CHECK: FH2
  return 0;
}
