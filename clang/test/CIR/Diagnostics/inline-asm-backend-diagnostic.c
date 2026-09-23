// REQUIRES: x86-registered-target
// RUN: not %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -S -o - %s 2>&1 | FileCheck %s

// CHECK: <inline asm>:1:10: error: unexpected token in argument list
// CHECK-NOT: LLVM ERROR
void foo(void) {
  __asm__ volatile("this is not an instruction");
}
