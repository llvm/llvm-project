// -O2
// RUN: %clang_cc1 -Wno-implicit-function-declaration  \
// RUN: -triple x86_64-unknown-unknown -fmath-errno -ffp-contract=on \
// RUN: -fno-rounding-math -O2 -emit-llvm -o - %s \
// RUN: | FileCheck %s

// -ffast-math
// RUN: %clang_cc1 -Wno-implicit-function-declaration  \
// RUN: -triple x86_64-unknown-unknown -menable-no-infs -menable-no-nans \
// RUN: -fapprox-func -funsafe-math-optimizations -fno-signed-zeros -mreassociate \
// RUN: -freciprocal-math -ffp-contract=fast -fno-rounding-math -ffast-math \
// RUN: -ffinite-math-only -ffast-math -emit-llvm -o - %s \
// RUN: | FileCheck %s -check-prefix=FAST

// -O0
// RUN: %clang_cc1 -Wno-implicit-function-declaration  \
// RUN: -triple x86_64-unknown-unknown -fmath-errno -ffp-contract=on \
// RUN: -fno-rounding-math -O0 \
// RUN: -emit-llvm -o - %s | FileCheck %s -check-prefix=NOOPT

#pragma float_control(precise,on)
float f1(float x) {
  return sqrtf(x);
}

// CHECK-LABEL: define {{.*}} float @f1
// CHECK: tail call float @sqrtf(float noundef {{.*}}) #[[ATTR4_O2:[0-9]+]]

// FAST-LABEL: define {{.*}} nofpclass(nan inf) float @f1
// FAST: call nofpclass(nan inf) float @sqrtf(float noundef nofpclass(nan inf) {{.*}}) #[[ATTR3_FAST:[0-9]+]]

// NOOPT-LABEL: define {{.*}} float @f1
// NOOPT: call float @sqrtf(float noundef {{.*}}) #[[ATTR4_NOOPT:[0-9]+]]

#pragma float_control(precise,off)
float f2(float x) {
  return sqrtf(x);
}

// CHECK-LABEL: define {{.*}} float @f2
// CHECK: tail call fast float @llvm.sqrt.f32(float {{.*}})

// FAST-LABEL: define {{.*}} nofpclass(nan inf) float @f2
// FAST: call fast float @llvm.sqrt.f32(float {{.*}})

// NOOPT-LABEL: define {{.*}} float @f2
// NOOPT: call fast float @sqrtf(float {{.*}}) #[[ATTR4_NOOPT:[0-9]+]]

__attribute__((optnone))
float f3(float x) {
  x = sqrtf(x);
  return x;
}

// CHECK-LABEL: define {{.*}} float @f3
// CHECK: call float @sqrtf(float noundef {{.*}})

// FAST-LABEL: define {{.*}} nofpclass(nan inf) float @f3
// FAST: call nofpclass(nan inf) float @sqrtf(float noundef nofpclass(nan inf) {{.*}}) #[[ATTR4_FAST:[0-9]+]]

// NOOPT-LABEL: define {{.*}} float @f3
// NOOPT:  call float @sqrtf(float noundef %0) #[[ATTR4_NOOPT:[0-9]+]]

// CHECK: [[ATTR4_O2]] = { nounwind }
// FAST: [[ATTR3_FAST]] =  { nounwind willreturn memory(none) }
// NOOPT: [[ATTR4_NOOPT]] = { nounwind }
