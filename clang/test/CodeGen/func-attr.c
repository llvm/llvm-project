// RUN: %clang -c -target x86_64 -ffast-math \
// RUN: -emit-llvm -S -o - %s | FileCheck %s

// RUN: %clang -c -target x86_64 -funsafe-math-optimizations \
// RUN: -emit-llvm -S -o - %s | FileCheck %s

float foo(float a, float b) {
  return a+b;
}

// CHECK: define{{.*}} float @foo(float noundef %{{.*}}, float noundef %{{.*}}){{.*}} [[FAST_ATTRS:#[0-9]+]]
// CHECK: attributes [[FAST_ATTRS]] = { {{.*}} "approx-func-fp-math"="true" {{.*}} "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" {{.*}} "unsafe-fp-math"="true"
