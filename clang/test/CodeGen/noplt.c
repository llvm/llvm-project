// RUN: %clang_cc1 -emit-llvm -fno-plt %s -o - | FileCheck %s -check-prefix=CHECK-NOPLT -check-prefix=CHECK-NOPLT-METADATA

// CHECK-NOPLT: Function Attrs: nonlazybind
// CHECK-NOPLT-NEXT: declare {{.*}}i32 @foo
// CHECK-NOPLT-METADATA: !"RtLibUseGOT"
int foo(void);

int bar(void) {
  return foo();
}
