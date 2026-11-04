// RUN: %clang_cc1 -fdenormal-fp-math=ieee %s -emit-llvm -o - | FileCheck %s --check-prefix=CHECK-IEEE
// RUN: %clang_cc1 -fdenormal-fp-math=preserve-sign %s -emit-llvm -o - | FileCheck %s --check-prefix=CHECK-PS
// RUN: %clang_cc1 -fdenormal-fp-math=positive-zero %s -emit-llvm -o - | FileCheck %s --check-prefix=CHECK-PZ
// RUN: %clang_cc1 -fdenormal-fp-math=dynamic %s -emit-llvm -o - | FileCheck %s --check-prefix=CHECK-DYNAMIC

// CHECK-LABEL: main

// The ieee,ieee is the default, so omit the attribute
// CHECK-IEEE-NOT:denormal_fpenv
// CHECK-PS: attributes #0 = {{.*}}denormal_fpenv(preservesign){{.*}}
// CHECK-PZ: attributes #0 = {{.*}}denormal_fpenv(positivezero){{.*}}
// CHECK-DYNAMIC: attributes #0 = {{.*}}denormal_fpenv(dynamic){{.*}}

int main(void) {
  return 0;
}
