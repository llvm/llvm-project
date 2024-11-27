// RUN: %clangxx_asan -O0 %s -o %t
// RUN: %run %t 2>&1
// CHECK: Success

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>

// On Windows, flags configured through the user-defined function `__asan_default_options`
// are suspected to not always be honored according to this GH issue:
// https://github.com/llvm/llvm-project/issues/117925
// This issue is resolved for the `allocator_may_return_null` flag, but not for all others.
// This test ensures we do not regress on `allocator_may_return_null` specifically.
extern "C" __declspec(dllexport) extern const char *__asan_default_options() {
  return "allocator_may_return_null=1";
}

int main() {
  // Attempt to allocate an excessive amount of memory, which should
  // terminate the program unless `allocator_may_return_null` is set.
  size_t max = std::numeric_limits<size_t>::max();

  free(malloc(max));
  printf("Success");
  return 0;
}
