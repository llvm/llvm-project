// RUN: %clangxx_asan -O0 %s -o %t
// RUN: %env_asan_opts=allocator_may_return_null=0 not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-ABORT
// RUN: %env_asan_opts=allocator_may_return_null=1 %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-RETURN-NULL

// RUN: %clangxx_asan -O0 %s -o %t -DUSER_FUNCTION
// RUN: %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-RETURN-NULL

#if USER_FUNCTION
// On Windows, flags configured through the user-defined function `__asan_default_options`
// are suspected to not always be honored according to GitHub bug:
// https://github.com/llvm/llvm-project/issues/117925
// This test ensures we do not regress on `allocator_may_return_null` specifically.
extern "C" __declspec(dllexport) extern const char *__asan_default_options() {
  return "allocator_may_return_null=1";
}
#endif

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>

int main() {
  // Attempt to allocate an excessive amount of memory, which should
  // terminate the program unless `allocator_may_return_null` is set.
  size_t max = std::numeric_limits<size_t>::max();

  // CHECK-ABORT: exceeds maximum supported size
  // CHECK-ABORT: ABORT
  free(malloc(max));

  printf("Success"); // CHECK-RETURN-NULL: Success
  return 0;
}
