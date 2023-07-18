// RUN: clang-extdef-mapping "%s" -- -fasm-blocks 2>&1 | FileCheck %s

void Break() {
  __asm { int 3 }
}

// CHECK: {{c:@F@Break .*}}
