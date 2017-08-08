// Test that AddressSanitizer does not re-animate dead globals when dead
// stripping is turned on.
//
// This test verifies that an out-of-bounds access on a global variable is
// detected after dead stripping has been performed. This proves that the
// runtime is able to register globals in the __DATA,__asan_globals section.

// REQUIRES: osx-ld64-live_support
// UNSUPPORTED: ios
// RUN: %clang_asan -mmacosx-version-min=10.11 -Xlinker -dead_strip -o %t %s
// RUN: llvm-nm -format=posix %t | FileCheck --check-prefix NM-CHECK %s
// RUN: not %run %t 2>&1 | FileCheck --check-prefix ASAN-CHECK %s

int alive[1] = {};
int dead[1] = {};
// NM-CHECK: {{^_alive }}
// NM-CHECK-NOT: {{^_dead }}

int main(int argc, char *argv[]) {
  alive[argc] = 0;
  // ASAN-CHECK: {{0x.* is located 0 bytes to the right of global variable}}
  return 0;
}
