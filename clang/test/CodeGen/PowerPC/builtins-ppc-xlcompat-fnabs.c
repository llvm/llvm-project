// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -triple powerpc64le-unknown-linux-gnu \
// RUN:   -emit-llvm %s -target-cpu pwr8 -o - | FileCheck %s
// RUN: %clang_cc1 -triple powerpc-unknown-aix \
// RUN:   -emit-llvm %s -target-cpu pwr8 -o - | FileCheck %s
// RUN: %clang_cc1 -triple powerpc64-unknown-linux-gnu \
// RUN:   -emit-llvm %s -target-cpu pwr7 -o - | FileCheck %s
// RUN: %clang_cc1 -triple powerpc64-unknown-aix \
// RUN:   -emit-llvm %s -target-cpu pwr7 -o - | FileCheck %s
// RUN: %clang_cc1 -triple powerpc64-unknown-linux-gnu \
// RUN:   -emit-llvm %s -target-cpu pwr6 -o - | FileCheck %s
// RUN: %clang_cc1 -triple powerpc64le-unknown-linux-gnu \
// RUN:   -emit-llvm %s -target-cpu pwr6 -o - | FileCheck %s
// RUN: %clang_cc1 -triple powerpc64-unknown-aix \
// RUN:   -emit-llvm %s -target-cpu pwr6 -o - | FileCheck %s
// RUN: %clang_cc1 -triple powerpc-unknown-aix \
// RUN:   -emit-llvm %s -target-cpu pwr6 -o - | FileCheck %s

extern float f;
extern double d;

// CHECK-LABEL: @test_fnabs(
// CHECK:       [[TMP0:%.*]] = load double, ptr @d
// CHECK-NEXT:  [[TMP1:%.*]] = call double @llvm.ppc.fnabs(double [[TMP0]])
// CHECK-NEXT:  ret double [[TMP1]]
double test_fnabs() {
  return __fnabs (d);
}

// CHECK-LABEL: @test_fnabss(
// CHECK:       [[TMP0:%.*]] = load float, ptr @f
// CHECK-NEXT:  [[TMP1:%.*]] = call float @llvm.ppc.fnabss(float [[TMP0]])
// CHECK-NEXT:  ret float [[TMP1]]
float test_fnabss() {
  return __fnabss (f);
}
