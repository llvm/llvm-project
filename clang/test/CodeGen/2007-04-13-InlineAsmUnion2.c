// RUN: %clang_cc1 %s -emit-llvm -o - | FileCheck %s

union U { int x; char* p; };
void foo() {
  union U bar;
  // CHECK: call void asm
  __asm__ volatile("foo %0\n" :: "r"(bar));
}
