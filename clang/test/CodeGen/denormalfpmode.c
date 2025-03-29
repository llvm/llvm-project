// RUN: %clang_cc1 -fdenormal-fp-math=ieee %s -emit-llvm -o - | FileCheck %s --check-prefix=CHECK-IEEE
// RUN: %clang_cc1 -fdenormal-fp-math=preserve-sign %s -emit-llvm -o - | FileCheck %s --check-prefix=CHECK-PS
// RUN: %clang_cc1 -fdenormal-fp-math=positive-zero %s -emit-llvm -o - | FileCheck %s --check-prefix=CHECK-PZ
// RUN: %clang_cc1 -fdenormal-fp-math=dynamic %s -emit-llvm -o - | FileCheck %s --check-prefix=CHECK-DYNAMIC

// CHECK-LABEL: main

// The ieee,ieee is the default, so omit the attribute
// CHECK-IEEE-NOT:"denormal-fp-math"
// CHECK-PS: attributes #0 = {{.*}}"denormal-fp-math"="preserve-sign,preserve-sign"{{.*}}
// CHECK-PZ: attributes #0 = {{.*}}"denormal-fp-math"="positive-zero,positive-zero"{{.*}}
// CHECK-DYNAMIC: attributes #0 = {{.*}}"denormal-fp-math"="dynamic,dynamic"{{.*}}

int main(void) {
  return 0;
}
