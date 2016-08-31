// RUN: %clang_cc1 -S -fdenormal-fp-math=ieee %s -emit-llvm -o - | FileCheck %s --check-prefix=CHECK-IEEE
// RUN: %clang_cc1 -S -fdenormal-fp-math=preserve-sign %s -emit-llvm -o - | FileCheck %s --check-prefix=CHECK-PS
// RUN: %clang_cc1 -S -fdenormal-fp-math=positive-zero %s -emit-llvm -o - | FileCheck %s --check-prefix=CHECK-PZ

// CHECK-LABEL: main
// CHECK-IEEE: attributes #0 = {{.*}}"denormal-fp-math"="ieee"{{.*}}
// CHECK-PS: attributes #0 = {{.*}}"denormal-fp-math"="preserve-sign"{{.*}}
// CHECK-PZ: attributes #0 = {{.*}}"denormal-fp-math"="positive-zero"{{.*}}

int main() {
  return 0;
}
