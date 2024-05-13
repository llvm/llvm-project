// Tests -fsanitize-coverage=control-flow.

// REQUIRES: has_sancovcc,stable-runtime
// UNSUPPORTED: i386-darwin, x86_64-darwin

// RUN: %clangxx -O0 -std=c++11 -fsanitize-coverage=control-flow %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s

#include <cstdint>
#include <cstdio>
#if __has_feature(ptrauth_calls)
  #include <ptrauth.h>
#else
  #define ptrauth_strip(__value, __key) (__value)
#endif

uintptr_t *CFS_BEG, *CFS_END;

extern "C" void __sanitizer_cov_cfs_init(const uintptr_t *cfs_beg,
                                         const uintptr_t *cfs_end) {
  CFS_BEG = (uintptr_t *)cfs_beg;
  CFS_END = (uintptr_t *)cfs_end;
}

__attribute__((noinline)) void foo(int x) { /* empty body */
}

void check_cfs_section(uintptr_t main_ptr, uintptr_t foo_ptr) {
  printf("Control Flow section boundaries: [%p %p)\n", CFS_BEG, CFS_END);
  uintptr_t *pt = CFS_BEG;
  uintptr_t currBB;

  while (pt < CFS_END) {
    currBB = *pt;
    pt++;

    if (currBB == main_ptr)
      printf("Saw the main().\n");
    else if (currBB == foo_ptr)
      printf("Saw the foo().\n");

    // Iterate over successors.
    while (*pt) {
      pt++;
    }
    pt++;
    // Iterate over callees.
    while (*pt) {
      if (*pt == foo_ptr && currBB != main_ptr)
        printf("Direct call matched.\n");
      if (*pt == -1 && currBB != main_ptr)
        printf("Indirect call matched.\n");
      pt++;
    }
    pt++;
  }
}

int main() {
  auto main_ptr = ptrauth_strip(&main, ptrauth_key_function_pointer);
  auto foo_ptr = ptrauth_strip(&foo, ptrauth_key_function_pointer);
  int x = 10;

  if (x > 0)
    foo(x);
  else
    (*foo_ptr)(x);

  check_cfs_section((uintptr_t)(*main_ptr), (uintptr_t)(*foo_ptr));

  printf("Finished!\n");
  return 0;
}

// CHECK: Control Flow section boundaries
// CHECK-DAG: Saw the foo().
// CHECK-DAG: Saw the main().
// CHECK-DAG: Direct call matched.
// CHECK-DAG: Indirect call matched.
// CHECK: Finished!
