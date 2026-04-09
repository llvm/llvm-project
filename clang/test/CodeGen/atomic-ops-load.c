// RUN: %clang_cc1 -triple riscv64 -O1 -emit-llvm %s -o - | FileCheck %s
#include <stdbool.h>

extern bool t1;
bool test1(void) {
// CHECK-LABEL: define{{.*}} i1 @test1
// CHECK: load atomic i8, ptr @t1 monotonic, align 1, !range ![[$WS_RANGE:[0-9]*]], !noundef !{{[0-9]+}}
// CHECK-NEXT: trunc nuw i8 %{{.*}} to i1
// CHECK-NEXT: ret i1 %{{.*}}
  return __atomic_load_n(&t1, __ATOMIC_RELAXED);
}
