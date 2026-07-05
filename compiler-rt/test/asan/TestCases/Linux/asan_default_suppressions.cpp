// Test that we use the suppressions from __asan_default_suppressions.
// RUN: %clangxx_asan %s -o %t && not %run %t 2>&1 | FileCheck %s
extern "C" {
  const char *__asan_default_suppressions() { return "FooBar"; }
}
// CHECK: AddressSanitizer: failed to parse suppressions
int main() {}
