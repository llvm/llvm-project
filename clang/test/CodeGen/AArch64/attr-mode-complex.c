// RUN: %clang_cc1 -triple arm64-none-linux-gnu -emit-llvm %s -o - | FileCheck %s

typedef _Complex float c16a __attribute((mode(HC)));
typedef _Complex double c16b __attribute((mode(HC)));
typedef _Complex float c32a __attribute((mode(SC)));
typedef _Complex double c32b __attribute((mode(SC)));
typedef _Complex float c64a __attribute((mode(DC)));
typedef _Complex double c64b __attribute((mode(DC)));

// CHECK: define{{.*}} { half, half } @c16_test([2 x half] noundef {{.*}}
// CHECK:   ret { half, half } {{.*}}
c16b c16_test(c16a x) {
  return x + x;
}

// CHECK: define{{.*}} { float, float } @c32_test([2 x float] noundef {{.*}})
// CHECK:   ret { float, float } {{.*}}
c32b c32_test(c32a x) {
  return x + x;
}

// CHECK: define{{.*}} { double, double } @c64_test([2 x double] noundef {{.*}})
// CHECK:   ret { double, double } {{.*}}
c64b c64_test(c64a x) {
  return x + x;
}
