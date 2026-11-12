// RUN: %clang_safestack -fno-stack-protector -D_FORTIFY_SOURCE=0 -g %s -o %t.nossp
// RUN: %run %t.nossp 2>&1 | FileCheck --check-prefix=NOSSP %s

// RUN: %clang_safestack -fstack-protector-all -D_FORTIFY_SOURCE=0 -g %s -o %t.ssp
// RUN: env LIBC_FATAL_STDERR_=1 not --crash %run %t.ssp 2>&1 | \
// RUN:     FileCheck -check-prefix=SSP %s

// Test stack canaries on the unsafe stack.

// REQUIRES: stable-runtime

#include <assert.h>
#include <stdio.h>
#include <string.h>

__attribute__((noinline)) void f(unsigned *y) {
  char x;
  char *volatile p = &x;
  char *volatile q = (char *)y;
  assert(p < q);
  assert(q - p < 1024); // sanity
  // This has technically undefined behavior, but we know the actual layout of
  // the unsafe stack and this should not touch anything important.
  memset(&x, 0xab, q - p + sizeof(*y));
}

int main(int argc, char **argv)
{
  unsigned y;
  // NOSSP: main 1
  // SSP: main 1
  fprintf(stderr, "main 1\n");
  f(&y);
  // NOSSP: main 2
  // SSP-NOT: main 2
  fprintf(stderr, "main 2\n");
  return 0;
}
