// RUN: %clangxx_asan -O0 %s -o %t
// RUN: %env_asan_opts=alloc_dealloc_mismatch=1 not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-MISMATCH
// RUN: %env_asan_opts=alloc_dealloc_mismatch=0 %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-SUCCESS

// RUN: %clangxx_asan -O0 %s -o %t -DUSER_FUNCTION
// RUN: not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-MISMATCH

// It is expected that ASAN_OPTS will override the value set through the user function.
// RUN: %env_asan_opts=alloc_dealloc_mismatch=0 %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-SUCCESS

#if USER_FUNCTION
// It's important to test the `alloc_dealloc_mismatch` flag set through the user function because, on Windows,
// flags configured through the user-defined function `__asan_default_options` are not always be honored.
// See: https://github.com/llvm/llvm-project/issues/117925
extern "C" __declspec(dllexport) extern const char *__asan_default_options() {
  return "alloc_dealloc_mismatch=1";
}
#endif

#include <cstdio>
#include <cstdlib>

// Tests the `alloc_dealloc_mismatch` flag set both via user function and through the environment variable.
int main() {
  // In the 'CHECK-MISMATCH' case, we simply check that the AddressSanitizer reports an error.
  delete (new int[10]); // CHECK-MISMATCH: AddressSanitizer:
  printf("Success");    // CHECK-SUCCESS: Success
  return 0;
}