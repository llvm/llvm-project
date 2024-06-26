// RUN: %clangxx -O2 %s -o %t && %run %t 2>&1 | FileCheck %s

// Malloc/free hooks are not supported on Windows.
// XFAIL: target={{.*windows-msvc.*}}

// Must not be implemented, no other reason to install interceptors.
// XFAIL: ubsan

#include <stdlib.h>
#include <unistd.h>
#include <sanitizer/allocator_interface.h>

extern "C" {
const volatile void *global_ptr;

#define WRITE(s) write(1, s, sizeof(s))

// Note: avoid calling functions that allocate memory in malloc/free
// to avoid infinite recursion.
void __sanitizer_malloc_hook(const volatile void *ptr, size_t sz) {
  if (__sanitizer_get_ownership(ptr) && sz == sizeof(int)) {
    WRITE("MallocHook\n");
    global_ptr = ptr;
  }
}
void __sanitizer_free_hook(const volatile void *ptr) {
  if (__sanitizer_get_ownership(ptr) && ptr == global_ptr)
    WRITE("FreeHook\n");
}
}  // extern "C"

volatile int *x;

void MallocHook1(const volatile void *ptr, size_t sz) { WRITE("MH1\n"); }
void MallocHook2(const volatile void *ptr, size_t sz) { WRITE("MH2\n"); }
void FreeHook1(const volatile void *ptr) { WRITE("FH1\n"); }
void FreeHook2(const volatile void *ptr) { WRITE("FH2\n"); }
// Call this function with uninitialized arguments to poison
// TLS shadow for function parameters before calling operator
// new and, eventually, user-provided hook.
__attribute__((noinline)) void allocate(int *unused1, int *unused2) {
  x = reinterpret_cast<int *>(malloc(sizeof(int)));
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
  if (global_ptr != (void*)x) {
    _exit(1);
  }

  // Check that realloc invokes hooks
  // We realloc to 128 here to avoid potential oversizing by allocators
  // making this a no-op.
  x = reinterpret_cast<int *>(realloc((int *)x, sizeof(int) * 128));
  // CHECK-DAG: FreeHook{{[[:space:]].*}}FH1{{[[:space:]].*}}FH2
  // CHECK-DAG: MH1{{[[:space:]].*}}MH2

  x[0] = 0;
  x[127] = -1;
  free((void *)x);
  // CHECK: FH1
  // CHECK: FH2
  return 0;
}
