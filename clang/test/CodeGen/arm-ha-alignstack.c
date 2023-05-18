// RUN: %clang_cc1 -triple armv7-eabi                                    -emit-llvm %s -o - | \
// RUN:     FileCheck %s --check-prefixes=CHECK,CHECK-SOFT
// RUN: %clang_cc1 -triple armv7-eabi -target-abi aapcs -mfloat-abi hard -emit-llvm %s -o - | \
// RUN:    FileCheck %s --check-prefixes=CHECK,CHECK-HARD
// REQUIRES: arm-registered-target

// CHECK: %struct.S0 = type { [4 x float] }
// CHECK: %struct.S1 = type { [2 x float] }
// CHECK: %struct.S2 = type { [4 x float] }
// CHECK: %struct.D0 = type { [2 x double] }
// CHECK: %struct.D1 = type { [2 x double] }
// CHECK: %struct.D2 = type { [4 x double] }

typedef struct {
  float v[4];
} S0;

float f0(S0 s) {
// CHECK-SOFT: define{{.*}} float @f0([4 x i32]  %s.coerce)
// CHECK-HARD: define{{.*}} arm_aapcs_vfpcc float @f0(%struct.S0 %s.coerce)
  return s.v[0];
}

float f0call() {
  S0 s = {0.0f, };
  return f0(s);
// CHECK-SOFT: call float @f0([4 x i32]
// CHECK-HARD: call arm_aapcs_vfpcc float @f0(%struct.S0
}

typedef struct {
  __attribute__((aligned(8))) float v[2];
} S1;

float f1(S1 s) {
// CHECK-SOFT: define{{.*}} float @f1([1 x i64]
// CHECK-HARD: define{{.*}} arm_aapcs_vfpcc float @f1(%struct.S1 alignstack(8)
  return s.v[0];
}

float f1call() {
  S1 s = {0.0f, };
  return f1(s);
// CHECK-SOFT: call float @f1([1 x i64
// CHECK-HARD: call arm_aapcs_vfpcc float @f1(%struct.S1 alignstack(8)
}

typedef struct {
  __attribute__((aligned(16))) float v[4];
} S2;

float f2(S2 s) {
// CHECK-SOFT: define{{.*}} float @f2([2 x i64]
// CHECK-HARD: define{{.*}} arm_aapcs_vfpcc float @f2(%struct.S2 alignstack(8)
  return s.v[0];
}

float f2call() {
  S2 s = {0.0f, };
  return f2(s);
// CHECK-SOFT: call float @f2([2 x i64]
// CHECK-HARD: call arm_aapcs_vfpcc float @f2(%struct.S2 alignstack(8)
}

typedef struct {
  double v[2];
} D0;

double g0(D0 d) {
// CHECK-SOFT: define{{.*}} double @g0([2 x i64]
// CHECK-HARD: define{{.*}} arm_aapcs_vfpcc double @g0(%struct.D0 %d.coerce
  return d.v[0];
}

double g0call() {
  D0 d = {0.0, };
  return g0(d);
// CHECK-SOFT: call double @g0([2 x i64]
// CHECK-HARD: call arm_aapcs_vfpcc double @g0(%struct.D0 %0
}

typedef struct {
  __attribute__((aligned(16))) double v[2];
} D1;

double g1(D1 d) {
// CHECK-SOFT: define{{.*}} double @g1([2 x i64]
// CHECK-HARD: define{{.*}} arm_aapcs_vfpcc double @g1(%struct.D1 alignstack(8)
  return d.v[0];
}

double g1call() {
  D1 d = {0.0, };
  return g1(d);
// CHECK-SOFT: call double @g1([2 x i64]
// CHECK-HARD: call arm_aapcs_vfpcc double @g1(%struct.D1 alignstack(8)
}

typedef struct {
  __attribute__((aligned(32))) double v[4];
} D2;

double g2(D2 d) {
// CHECK-SOFT: define{{.*}} double @g2([4 x i64]
// CHECK-HARD: define{{.*}} arm_aapcs_vfpcc double @g2(%struct.D2 alignstack(8)
  return d.v[0];
}

double g2call() {
  D2 d = {0.0, };
  return g2(d);
// CHECK-SOFT: call double @g2([4 x i64]
// CHECK-HARD: call arm_aapcs_vfpcc double @g2(%struct.D2 alignstack(8)
}
