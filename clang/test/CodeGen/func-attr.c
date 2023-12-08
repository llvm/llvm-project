// RUN: %clang_cc1 -triple x86_64-linux-gnu -ffast-math \
// RUN: -ffp-contract=fast -emit-llvm -o - %s | \
// RUN: FileCheck %s --check-prefixes=CHECK,CHECK-UNSAFE,FINITEONLY

// RUN: %clang_cc1 -triple x86_64-linux-gnu -funsafe-math-optimizations \
// RUN: -ffp-contract=fast -emit-llvm -o - %s | \
// RUN: FileCheck %s --check-prefixes=CHECK,CHECK-UNSAFE,NOFINITEONLY

// RUN: %clang_cc1 -triple x86_64-linux-gnu -funsafe-math-optimizations \
// RUN: -ffp-contract=on -emit-llvm -o - %s | \
// RUN: FileCheck %s --check-prefixes=CHECK,CHECK-NOUNSAFE,NOFINITEONLY

// RUN: %clang_cc1 -triple x86_64-linux-gnu -funsafe-math-optimizations \
// RUN: -ffp-contract=off -emit-llvm -o - %s | \
// RUN: FileCheck %s --check-prefixes=CHECK,CHECK-NOUNSAFE,NOFINITEONLY

float foo(float a, float b) {
  return a+b;
}

// FINITEONLY:    define{{.*}} float @foo(float noundef nofpclass(nan inf) %{{.*}}, float noundef nofpclass(nan inf) %{{.*}}){{.*}} [[ATTRS:#[0-9]+]]
// NOFINITEONLY:    define{{.*}} float @foo(float noundef %{{.*}}, float noundef %{{.*}}){{.*}} [[ATTRS:#[0-9]+]]

// CHECK:              attributes [[ATTRS]] = {
// CHECK-SAME:           "approx-func-fp-math"="true"
// CHECK-SAME:           "no-signed-zeros-fp-math"="true"
// CHECK-SAME:           "no-trapping-math"="true"
// CHECK-UNSAFE-SAME:    "unsafe-fp-math"="true"
// CHECK-NOUNSAFE-NOT:   "unsafe-fp-math"="true"
// CHECK-SAME:         }
