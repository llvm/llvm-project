// RUN: %clangxx_asan -O0 %s -o %t
// RUN: %env_asan_opts=malloc_context_size=0:fast_unwind_on_malloc=0 not %run %t 2>&1 | FileCheck %s
// RUN: %env_asan_opts=malloc_context_size=0:fast_unwind_on_malloc=1 not %run %t 2>&1 | FileCheck %s
// RUN: %env_asan_opts=malloc_context_size=1:fast_unwind_on_malloc=0 not %run %t 2>&1 | FileCheck %s
// RUN: %env_asan_opts=malloc_context_size=1:fast_unwind_on_malloc=1 not %run %t 2>&1 | FileCheck %s
// RUN: %env_asan_opts=malloc_context_size=2 not %run %t 2>&1 | FileCheck %s --check-prefix=TWO

int main() {
  char *x = new char[20];
  delete[] x;
  return x[0];

  // CHECK: freed by thread T{{.*}} here:
  // CHECK-NEXT: #0 0x{{.*}} in {{operator delete( )?\[\]|wrap__ZdaPv}}
  // CHECK-NOT: #1 0x{{.*}}

  // CHECK: previously allocated by thread T{{.*}} here:
  // CHECK-NEXT: #0 0x{{.*}} in {{operator new( )?\[\]|wrap__Znam}}
  // CHECK-NOT: #1 0x{{.*}}

  // CHECK: SUMMARY: AddressSanitizer: heap-use-after-free

  // TWO: previously allocated by thread T{{.*}} here:
  // TWO-NEXT: #0 0x{{.*}}
  // TWO-NEXT: #1 0x{{.*}} in main {{.*}}malloc_context_size.cpp
  // TWO: SUMMARY: AddressSanitizer: heap-use-after-free
}
