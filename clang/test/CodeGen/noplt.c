// RUN: %clang_cc1 -emit-llvm -fno-plt %s -o - | FileCheck %s -check-prefix=CHECK-NOPLT

// CHECK-NOPLT: Function Attrs: nonlazybind
// CHECK-NOPLT-NEXT: declare {{.*}}i32 @foo
int foo();

int bar() {
  return foo();
}
