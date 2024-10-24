// RUN: %clangxx_asan -O2 %s -o %t -DTEST=basic_hook_works && not %run %t \
// RUN:   2>&1 | FileCheck %s -check-prefix=CHECK-BASIC
// RUN: %clangxx_asan -O2 %s -o %t -DTEST=ignore && %run %t \
// RUN:   2>&1 | FileCheck %s -check-prefix=CHECK-IGNORE
// RUN: %clangxx_asan -O2 %s -o %t -DTEST=ignore_twice && not %run %t \
// RUN:   2>&1 | FileCheck %s -check-prefix=CHECK-IGNORE-2
// RUN: %clangxx_asan -O2 %s -o %t -DTEST=mismatch && %env_asan_opts=alloc_dealloc_mismatch=1 not %run %t \
// RUN:   2>&1 | FileCheck %s -check-prefix=CHECK-MISMATCH
// RUN: %clangxx_asan -O2 %s -o %t -DTEST=ignore_mismatch && %env_asan_opts=alloc_dealloc_mismatch=1 %run %t \
// RUN:   2>&1 | FileCheck %s -check-prefix=CHECK-IGNORE-MISMATCH
// RUN: %clangxx_asan -O2 %s -o %t -DTEST=double_delete && not %run %t \
// RUN:   2>&1 | FileCheck %s -check-prefix=CHECK-DOUBLE-DELETE

#include <stdio.h>
#include <stdlib.h>

static char *volatile glob_ptr;
bool ignore_free = false;

#if (__APPLE__)
// Required for dyld macOS 12.0+
#  define WEAK_ON_APPLE __attribute__((weak))
#else // !(__APPLE__)
#  define WEAK_ON_APPLE
#endif // (__APPLE__)

extern "C" {
WEAK_ON_APPLE void __sanitizer_free_hook(const volatile void *ptr) {
  if (ptr == glob_ptr)
    fprintf(stderr, "Free Hook\n");
}

WEAK_ON_APPLE int __sanitizer_ignore_free_hook(const volatile void *ptr) {
  if (ptr != glob_ptr)
    return 0;
  fprintf(stderr, ignore_free ? "Free Ignored\n" : "Free Respected\n");
  return ignore_free;
}
} // extern "C"

void allocate() { glob_ptr = reinterpret_cast<char *volatile>(malloc(100)); }
void deallocate() { free(reinterpret_cast<void *>(glob_ptr)); }

void basic_hook_works() {
  allocate();
  deallocate();  // CHECK-BASIC-NOT: Free Ignored
                 // CHECK-BASIC:     Free Respected
                 // CHECK-BASIC:     Free Hook
  *glob_ptr = 0; // CHECK-BASIC:     AddressSanitizer: heap-use-after-free
}

void ignore() {
  allocate();
  ignore_free = true;
  deallocate();
  // CHECK-IGNORE:     Free Ignored
  // CHECK-IGNORE-NOT: Free Respected
  // CHECK-IGNORE-NOT: Free Hook
  // CHECK-IGNORE-NOT: AddressSanitizer
  *glob_ptr = 0;
}

void ignore_twice() {
  allocate();
  ignore_free = true;
  deallocate(); // CHECK-IGNORE-2: Free Ignored
  *glob_ptr = 0;
  ignore_free = false;
  deallocate();  // CHECK-IGNORE-2-NOT: Free Ignored
                 // CHECK-IGNORE-2:     Free Respected
                 // CHECK-IGNORE-2:     Free Hook
  *glob_ptr = 0; // CHECK-IGNORE-2:     AddressSanitizer: heap-use-after-free
}

void ignore_a_lot() {
  allocate();
  ignore_free = true;
  for (int i = 0; i < 10000; ++i) {
    deallocate(); // CHECK-IGNORE-3: Free Ignored
    *glob_ptr = 0;
  }
  ignore_free = false;
  deallocate();  // CHECK-IGNORE-3: Free Respected
                 // CHECK-IGNORE-3: Free Hook
  *glob_ptr = 0; // CHECK-IGNORE-3: AddressSanitizer: heap-use-after-free
}

void mismatch() {
  glob_ptr = new char;
  deallocate(); // CHECK-MISMATCH: AddressSanitizer: alloc-dealloc-mismatch
}

void ignore_mismatch() {
  glob_ptr = new char;
  ignore_free = true;
  // Mismatch isn't detected when the free() is ignored.
  deallocate();
  deallocate();
  ignore_free = false;
  // And also isn't detected when the memory is free()-d for real.
  deallocate(); // CHECK-IGNORE-MISMATCH-NOT: AddressSanitizer: alloc-dealloc-mismatch
}

void double_delete() {
  allocate();
  ignore_free = true;
  deallocate(); // CHECK-DOUBLE-DELETE: Free Ignored
  deallocate(); // CHECK-DOUBLE-DELETE: Free Ignored
  ignore_free = false;
  deallocate(); // CHECK-DOUBLE-DELETE: Free Respected
                // CHECK-DOUBLE-DELETE: Free Hook
  deallocate(); // CHECK-DOUBLE-DELETE: AddressSanitizer: attempting double-free
}

int main() {
  TEST();
  return 0;
}
