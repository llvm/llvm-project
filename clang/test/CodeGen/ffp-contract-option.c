// REQUIRES: x86-registered-target
// RUN: %clang_cc1 -no-opaque-pointers -triple=x86_64 %s -emit-llvm -o - \
// RUN:| FileCheck --check-prefixes CHECK,CHECK-DEFAULT  %s

// RUN: %clang_cc1 -no-opaque-pointers -triple=x86_64 -ffp-contract=off %s -emit-llvm -o - \
// RUN:| FileCheck --check-prefixes CHECK,CHECK-DEFAULT  %s

// RUN: %clang_cc1 -no-opaque-pointers -triple=x86_64 -ffp-contract=on %s -emit-llvm -o - \
// RUN:| FileCheck --check-prefixes CHECK,CHECK-ON  %s

// RUN: %clang_cc1 -no-opaque-pointers -triple=x86_64 -ffp-contract=fast %s -emit-llvm -o - \
// RUN:| FileCheck --check-prefixes CHECK,CHECK-CONTRACTFAST  %s

// RUN: %clang_cc1 -no-opaque-pointers -triple=x86_64 -ffast-math %s -emit-llvm -o - \
// RUN:| FileCheck --check-prefixes CHECK,CHECK-CONTRACTOFF %s

// RUN: %clang_cc1 -no-opaque-pointers -triple=x86_64 -ffast-math -ffp-contract=off %s -emit-llvm \
// RUN: -o - | FileCheck --check-prefixes CHECK,CHECK-CONTRACTOFF %s

// RUN: %clang_cc1 -no-opaque-pointers -triple=x86_64 -ffast-math -ffp-contract=on %s -emit-llvm \
// RUN: -o - | FileCheck --check-prefixes CHECK,CHECK-ONFAST %s

// RUN: %clang_cc1 -no-opaque-pointers -triple=x86_64 -ffast-math -ffp-contract=fast %s -emit-llvm \
// RUN:  -o - | FileCheck --check-prefixes CHECK,CHECK-FASTFAST %s

// RUN: %clang_cc1 -no-opaque-pointers -triple=x86_64 -ffp-contract=fast -ffast-math  %s \
// RUN: -emit-llvm \
// RUN:  -o - | FileCheck --check-prefixes CHECK,CHECK-FASTFAST %s

// RUN: %clang_cc1 -no-opaque-pointers -triple=x86_64 -ffp-contract=off -fmath-errno \
// RUN: -fno-rounding-math %s -emit-llvm -o - \
// RUN:  -o - | FileCheck --check-prefixes CHECK,CHECK-NOFAST %s

// RUN: %clang -S -emit-llvm -fno-fast-math %s -o - \
// RUN: | FileCheck %s --check-prefixes=CHECK,CHECK-FPC-ON

// RUN: %clang -S -emit-llvm -ffp-contract=fast -fno-fast-math \
// RUN: %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-FPC-ON

// RUN: %clang -S -emit-llvm -ffp-contract=on -fno-fast-math \
// RUN: %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-FPC-ON

// RUN: %clang -S -emit-llvm -ffp-contract=off -fno-fast-math \
// RUN: %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-FPC-OFF

// RUN: %clang -S -emit-llvm -ffp-model=fast -fno-fast-math \
// RUN: %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-FPC-ON

// RUN: %clang -S -emit-llvm -ffp-model=precise -fno-fast-math \
// RUN: %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-FPC-ON

// RUN: %clang -S -emit-llvm -ffp-model=strict -fno-fast-math \
// RUN: -target x86_64 %s -o - | FileCheck %s \
// RUN: --check-prefixes=CHECK,CHECK-FPSC-OFF

// RUN: %clang -S -emit-llvm -ffast-math -fno-fast-math \
// RUN: %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-FPC-ON

float mymuladd(float x, float y, float z) {
  // CHECK: define{{.*}} float @mymuladd
  return x * y + z;
  // expected-warning{{overriding '-ffp-contract=fast' option with '-ffp-contract=on'}}

  // CHECK-DEFAULT: load float, float*
  // CHECK-DEFAULT: fmul float
  // CHECK-DEFAULT: load float, float*
  // CHECK-DEFAULT: fadd float

  // CHECK-ON: load float, float*
  // CHECK-ON: load float, float*
  // CHECK-ON: load float, float*
  // CHECK-ON: call float @llvm.fmuladd.f32(float {{.*}}, float {{.*}}, float {{.*}})

  // CHECK-CONTRACTFAST: load float, float*
  // CHECK-CONTRACTFAST: load float, float*
  // CHECK-CONTRACTFAST: fmul contract float
  // CHECK-CONTRACTFAST: load float, float*
  // CHECK-CONTRACTFAST: fadd contract float

  // CHECK-CONTRACTOFF: load float, float*
  // CHECK-CONTRACTOFF: load float, float*
  // CHECK-CONTRACTOFF: fmul reassoc nnan ninf nsz arcp afn float
  // CHECK-CONTRACTOFF: load float, float*
  // CHECK-CONTRACTOFF: fadd reassoc nnan ninf nsz arcp afn float {{.*}}, {{.*}}

  // CHECK-ONFAST: load float, float*
  // CHECK-ONFAST: load float, float*
  // CHECK-ONFAST: load float, float*
  // CHECK-ONFAST: call reassoc nnan ninf nsz arcp afn float @llvm.fmuladd.f32(float {{.*}}, float {{.*}}, float {{.*}})

  // CHECK-FASTFAST: load float, float*
  // CHECK-FASTFAST: load float, float*
  // CHECK-FASTFAST: fmul fast float
  // CHECK-FASTFAST: load float, float*
  // CHECK-FASTFAST: fadd fast float {{.*}}, {{.*}}

  // CHECK-NOFAST: load float, float*
  // CHECK-NOFAST: load float, float*
  // CHECK-NOFAST: fmul float {{.*}}, {{.*}}
  // CHECK-NOFAST: load float, float*
  // CHECK-NOFAST: fadd float {{.*}}, {{.*}}

  // CHECK-FPC-ON: load float, float*
  // CHECK-FPC-ON: load float, float*
  // CHECK-FPC-ON: load float, float*
  // CHECK-FPC-ON: call float @llvm.fmuladd.f32(float {{.*}}, float {{.*}}, float {{.*}})

  // CHECK-FPC-OFF: load float, float*
  // CHECK-FPC-OFF: load float, float*
  // CHECK-FPC-OFF: fmul float
  // CHECK-FPC-OFF: load float, float*
  // CHECK-FPC-OFF: fadd float {{.*}}, {{.*}}

  // CHECK-FFPC-OFF: load float, float*
  // CHECK-FFPC-OFF: load float, float*
  // CHECK-FPSC-OFF: call float @llvm.experimental.constrained.fmul.f32(float {{.*}}, float {{.*}}, {{.*}})
  // CHECK-FPSC-OFF: load float, float*
  // CHECK-FPSC-OFF: [[RES:%.*]] = call float @llvm.experimental.constrained.fadd.f32(float {{.*}}, float {{.*}}, {{.*}})

}
