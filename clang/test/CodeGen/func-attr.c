// RUN: %clang_cc1 -triple x86_64-linux-gnu -ffast-math \
// RUN: -ffp-contract=fast -emit-llvm -o - %s | \
// RUN: FileCheck %s --check-prefixes=CHECK,CHECK-UNSAFE

// RUN: %clang_cc1 -triple x86_64-linux-gnu -funsafe-math-optimizations \
// RUN: -ffp-contract=fast -emit-llvm -o - %s | \
// RUN: FileCheck %s --check-prefixes=CHECK,CHECK-UNSAFE

// RUN: %clang_cc1 -triple x86_64-linux-gnu -funsafe-math-optimizations \
// RUN: -ffp-contract=on -emit-llvm -o - %s | \
// RUN: FileCheck %s --check-prefixes=CHECK,CHECK-NOUNSAFE

// RUN: %clang_cc1 -triple x86_64-linux-gnu -funsafe-math-optimizations \
// RUN: -ffp-contract=off -emit-llvm -o - %s | \
// RUN: FileCheck %s --check-prefixes=CHECK,CHECK-NOUNSAFE

float foo(float a, float b) {
  return a+b;
}

// CHECK:              define{{.*}} float @foo(float noundef %{{.*}}, float noundef %{{.*}}){{.*}} [[ATTRS:#[0-9]+]]
// CHECK:              attributes [[ATTRS]] = {
// CHECK-SAME:           "approx-func-fp-math"="true"
// CHECK-SAME:           "no-signed-zeros-fp-math"="true"
// CHECK-SAME:           "no-trapping-math"="true"
// CHECK-UNSAFE-SAME:    "unsafe-fp-math"="true"
// CHECK-NOUNSAFE-NOT:   "unsafe-fp-math"="true"
// CHECK-SAME:         }
