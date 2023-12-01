// RUN: %clangxx_asan -O0 %s -o %t
// RUN: %env_asan_opts=malloc_context_size=0:fast_unwind_on_malloc=0 not %run %t 2>&1 | FileCheck %s --check-prefixes=CHECK,%if target={{.*-windows-.*}} %{CHECK-WIN%} %else %{CHECK-NOT-WIN%}
// RUN: %env_asan_opts=malloc_context_size=0:fast_unwind_on_malloc=1 not %run %t 2>&1 | FileCheck %s --check-prefixes=CHECK,%if target={{.*-windows-.*}} %{CHECK-WIN%} %else %{CHECK-NOT-WIN%}
// RUN: %env_asan_opts=malloc_context_size=1:fast_unwind_on_malloc=0 not %run %t 2>&1 | FileCheck %s --check-prefixes=CHECK,%if target={{.*-windows-.*}} %{CHECK-WIN%} %else %{CHECK-NOT-WIN%}
// RUN: %env_asan_opts=malloc_context_size=1:fast_unwind_on_malloc=1 not %run %t 2>&1 | FileCheck %s --check-prefixes=CHECK,%if target={{.*-windows-.*}} %{CHECK-WIN%} %else %{CHECK-NOT-WIN%}
// RUN: %env_asan_opts=malloc_context_size=2 not %run %t 2>&1 | FileCheck %s --check-prefixes=TWO,%if target={{.*-windows-.*}} %{TWO-WIN%} %else %{TWO-NOT-WIN%}

int main() {
  char *x = new char[20];
  delete[] x;
  return x[0];

  // CHECK: freed by thread T{{.*}} here:
  // CHECK-NOT-WIN-NEXT: #0 0x{{.*}} in {{operator delete( )?\[\]|_ZdaPv}}
  // CHECK-WIN-NEXT: #0 0x{{.*}} in {{__asan_delete_array}}
  // CHECK-NOT: #1 0x{{.*}}

  // CHECK: previously allocated by thread T{{.*}} here:
  // CHECK-NOT-WIN-NEXT: #0 0x{{.*}} in {{operator new( )?\[\]|_Znam}}
  // CHECK-WIN-NEXT: #0 0x{{.*}} in {{__asan_new_array}}
  // CHECK-NOT: #1 0x{{.*}}

  // CHECK: SUMMARY: AddressSanitizer: heap-use-after-free

  // TWO: previously allocated by thread T{{.*}} here:
  // TWO-NEXT: #0 0x{{.*}}
  // TWO-NOT-WIN-NEXT: #1 0x{{.*}} in main {{.*}}malloc_context_size.cpp
  // TWO-WIN-NEXT: #1 0x{{.*}} in {{operator new( )?\[\]|_Znam}}
  // TWO: SUMMARY: AddressSanitizer: heap-use-after-free
}
