// RUN: %clangxx -fsanitize=realtime %s -o %t
// RUN: not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx %s -fsanitize=realtime -o - -S -emit-llvm | FileCheck %s --check-prefix=CHECK-ENABLED-IR
// RUN: %clangxx %s -o - -S -emit-llvm | FileCheck %s --check-prefix=CHECK-DISABLED-IR
// UNSUPPORTED: ios

#include <stdio.h>
#include <stdlib.h>

#include "sanitizer/rtsan_interface.h"

void violation() [[clang::nonblocking]] {
  void *ptr;
  {
    __rtsan::ScopedDisabler disabler{};
    ptr = malloc(2);
    fprintf(stderr, "Allocated pointer %p in disabled context\n", ptr);
  }

  // ensure nested disablers don't interfere with one another
  {
    void *ptr2;
    __rtsan::ScopedDisabler disabler{};
    {
      __rtsan::ScopedDisabler disabler2{};
      ptr2 = malloc(2);
      fprintf(stderr, "Allocated second pointer %p in disabled context\n",
              ptr2);
    }

    free(ptr2);
    fprintf(stderr, "Free'd second pointer in disabled context\n");
  }

  free(ptr);
}

int main() {
  violation();
  return 0;
  // CHECK: Allocated pointer {{.*}} in disabled context
  // CHECK: Allocated second pointer {{.*}} in disabled context
  // CHECK: Free'd second pointer in disabled context
  // CHECK: ==ERROR: RealtimeSanitizer: unsafe-library-call
  // CHECK-NOT: {{.*malloc*}}
  // CHECK-NEXT: {{.*free.*}}
}

// CHECK-ENABLED-IR: {{.*@__rtsan_disable.*}}
// CHECK-ENABLED-IR: {{.*@__rtsan_enable.*}}

// CHECK-DISABLED-IR-NOT: {{.*__rtsan_disable.*}}
// CHECK-DISABLED-IR-NOT: {{.*__rtsan_enable.*}}
