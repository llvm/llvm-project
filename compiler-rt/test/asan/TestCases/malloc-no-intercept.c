// Test that on non-glibc platforms, a number of malloc-related functions are
// not intercepted.

// RUN: not %clang_asan -Dtestfunc=mallinfo %s -o %t
// RUN: not %clang_asan -Dtestfunc=mallopt  %s -o %t
// RUN: not %clang_asan -Dtestfunc=memalign %s -o %t
// RUN: not %clang_asan -Dtestfunc=pvalloc  %s -o %t
// RUN: not %clang_asan -Dtestfunc=cfree    %s -o %t

// Conflicts with BIONIC declarations.
// UNSUPPORTED: android
// Lacks mallinfo, mallopt except in libmalloc.  cfree with different
// signature in libc.
// UNSUPPORTED: solaris

// Inhibit conflicting declaration of memalign on Solaris.
#if defined(__sun__) && defined(__svr4__)
#undef __EXTENSIONS__
#endif

#include <stdlib.h>

// For glibc, cause link failures by referencing a nonexistent function.
#ifdef __GLIBC__
#undef testfunc
#define testfunc nonexistent_function
#endif

void testfunc(void);

int main(void)
{
  testfunc();
  return 0;
}
