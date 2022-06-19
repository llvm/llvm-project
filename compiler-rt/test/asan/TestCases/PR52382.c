// RUN: %clang_asan -O0 %s -o %t -w && not %run %t 2>&1 | FileCheck %s
// RUN: %clang_asan -O2 %s -o %t -w && not %run %t 2>&1 | FileCheck %s

int global_array[100] = {-1};

// This access is 412 bytes after the start of the global: past the end of the
// uninstrumented array, but within the bounds of the extended instrumented
// array. We should ensure this is still instrumented.
int main(void) { return global_array[103]; }

// CHECK: AddressSanitizer: global-buffer-overflow on address
// CHECK: is located 12 bytes to the right of global variable 'global_array'
