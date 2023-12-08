// RUN: clang-extdef-mapping "%s" -- -fasm-blocks \
// RUN:   -target x86_64-apple-darwin10 2>&1 | FileCheck %s
// REQUIRES: x86-registered-target

void Break() {
  __asm { int 3 }
}

// CHECK: {{10:c:@F@Break}}
