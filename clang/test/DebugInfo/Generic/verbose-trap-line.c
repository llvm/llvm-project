// REQUIRES: x86-registered-target
// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-obj -debug-info-kind=limited %s -o %t
// RUN: llvm-objdump -d -l %t | FileCheck %s

void test_trap_function(void) {
  int x = 1;

// CHECK-LABEL: <test_trap_function>:
// CHECK: verbose-trap-line.c:[[# @LINE+2]]
// CHECK-NEXT: {{.*}}ud2
  __builtin_verbose_trap("category", "message");
}
