// RUN: %clang -c -ffast-math -emit-llvm -S -o - %s \
// RUN: | FileCheck %s

// RUN: %clang -c -funsafe-math-optimizations -emit-llvm -S -o - %s \
// RUN: | FileCheck %s

float foo(float a, float b) {
  return a+b;
}

// CHECK: define{{.*}} float @foo(float noundef %{{.*}}, float noundef %{{.*}}) [[FAST_ATTRS:#[0-9]+]]
// CHECK: attributes [[FAST_ATTRS]] = { {{.*}} "approx-func-fp-math"="true" {{.*}} "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" {{.*}} "unsafe-fp-math"="true"
