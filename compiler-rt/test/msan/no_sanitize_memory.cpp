// RUN: %clangxx_msan -O0 %s -o %t && %run %t >%t.out 2>&1
// RUN: %clangxx_msan -O1 %s -o %t && %run %t >%t.out 2>&1
// RUN: %clangxx_msan -O2 %s -o %t && %run %t >%t.out 2>&1
// RUN: %clangxx_msan -O3 %s -o %t && %run %t >%t.out 2>&1

// RUN: %clangxx_msan -O0 %s -o %t -DCHECK_IN_F && %run %t >%t.out 2>&1
// RUN: %clangxx_msan -O1 %s -o %t -DCHECK_IN_F && %run %t >%t.out 2>&1
// RUN: %clangxx_msan -O2 %s -o %t -DCHECK_IN_F && %run %t >%t.out 2>&1
// RUN: %clangxx_msan -O3 %s -o %t -DCHECK_IN_F && %run %t >%t.out 2>&1

// Test that (no_sanitize_memory) functions
// * don't check shadow values (-DCHECK_IN_F)
// * treat all values loaded from memory as fully initialized (-UCHECK_IN_F)

#include <stdlib.h>
#include <stdio.h>

__attribute__((noinline))
__attribute__((no_sanitize_memory))
int f(void) {
  int x;
  int * volatile p = &x;
#ifdef CHECK_IN_F
  if (*p)
    exit(0);
#endif
  return *p;
}

int main(void) {
  if (f())
    exit(0);
  return 0;
}
