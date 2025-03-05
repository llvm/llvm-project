// RUN: %clang_cc1 -triple arm64-none-linux-gnu -emit-llvm %s -o - | FileCheck %s

typedef float f16a __attribute((mode(HF)));
typedef double f16b __attribute((mode(HF)));
typedef float f32a __attribute((mode(SF)));
typedef double f32b __attribute((mode(SF)));
typedef float f64a __attribute((mode(DF)));
typedef double f64b __attribute((mode(DF)));
f16b tmp;

// CHECK: define{{.*}} ptr @f16_test(ptr noundef {{.*}})
// CHECK:   store half {{.*}}, ptr @tmp, align 2
// CHECK:   ret ptr @tmp
f16b *f16_test(f16a *x) {
  tmp = *x + *x;
  return &tmp;
}

// CHECK: define{{.*}} float @f32_test(float noundef {{.*}})
// CHECK:   ret float {{.*}}
f32b f32_test(f32a x) {
  return x + x;
}

// CHECK: define{{.*}} double @f64_test(double noundef {{.*}})
// CHECK:   ret double {{.*}}
f64b f64_test(f64a x) {
  return x + x;
}
