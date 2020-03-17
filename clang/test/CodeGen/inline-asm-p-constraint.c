// RUN: %clang_cc1 -emit-llvm %s -o - -triple aarch64-linux-gnu | FileCheck %s
void foo(void* ptr) {
  asm volatile("prfm pldl1keep, %a0\n" : : "p" (ptr));
  // CHECK:     call void asm sideeffect "prfm pldl1keep, ${0:a}\0A", "r"(i8* %0)
  // CHECK-NOT: call void asm sideeffect "prfm pldl1keep, ${0:a}\0A", "p"(i8* %0)
}
