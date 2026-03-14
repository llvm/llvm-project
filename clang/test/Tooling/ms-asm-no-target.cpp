// RUN: clang-check "%s" -- -fasm-blocks -target x86_64-apple-darwin10 2>&1 | FileCheck -check-prefix=CHECK-X86 %s -allow-empty
// REQUIRES: x86-registered-target

void Break() {
  __asm { int 3 }
}

// clang-check should initialize the x86 target, so x86 should work.
// CHECK-X86-NOT: error: MS-style inline assembly is not available
