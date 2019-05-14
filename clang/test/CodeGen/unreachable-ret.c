// RUN: %clang_cc1 -emit-llvm -o - %s | FileCheck %s

extern void abort() __attribute__((noreturn));

void f1() {
  abort();
}
// CHECK-LABEL: define {{.*}}void @f1()
// CHECK-NEXT: entry:
// CHECK-NEXT:   call void @abort()
// CHECK-NEXT:   unreachable
// CHECK-NEXT: }

void *f2() {
  abort();
  return 0;
}
// CHECK-LABEL: define {{.*}}i8* @f2()
// CHECK-NEXT: entry:
// CHECK-NEXT:   call void @abort()
// CHECK-NEXT:   unreachable
// CHECK-NEXT: }

