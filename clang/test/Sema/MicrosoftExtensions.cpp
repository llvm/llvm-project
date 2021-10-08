// RUN: %clang_cc1 -triple i686-windows %s -fsyntax-only -Wmicrosoft -verify -fms-extensions
// RUN: %clang_cc1 -triple x86_64-windows %s -fsyntax-only -Wmicrosoft -verify -fms-extensions
// expected-no-diagnostics

// Check that __ptr32/__ptr64 can be compared.
int test_ptr_comparison(int *__ptr32 __uptr p32u, int *__ptr32 __sptr p32s,
                        int *__ptr64 p64) {
  return (p32u == p32s) +
         (p32u == p64) +
         (p32s == p64);
}
