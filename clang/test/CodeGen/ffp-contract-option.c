// REQUIRES: x86-registered-target
// UNSUPPORTED: target={{.*}}-zos{{.*}}
// RUN: %clang_cc1 -triple=x86_64 %s -emit-llvm -o - \
// RUN:| FileCheck --check-prefixes CHECK,CHECK-DEFAULT  %s

// RUN: %clang_cc1 -triple=x86_64 -ffp-contract=off %s -emit-llvm -o - \
// RUN:| FileCheck --check-prefixes CHECK,CHECK-DEFAULT  %s

// RUN: %clang_cc1 -triple=x86_64 -ffp-contract=on %s -emit-llvm -o - \
// RUN:| FileCheck --check-prefixes CHECK,CHECK-ON  %s

// RUN: %clang_cc1 -triple=x86_64 -ffp-contract=fast %s -emit-llvm -o - \
// RUN:| FileCheck --check-prefixes CHECK,CHECK-CONTRACTFAST  %s

// RUN: %clang_cc1 -triple=x86_64 -ffast-math %s -emit-llvm -o - \
// RUN:| FileCheck --check-prefixes CHECK,CHECK-CONTRACTOFF %s

// RUN: %clang_cc1 -triple=x86_64 -ffast-math -ffp-contract=off %s -emit-llvm \
// RUN: -o - | FileCheck --check-prefixes CHECK,CHECK-CONTRACTOFF %s

// RUN: %clang_cc1 -triple=x86_64 -ffast-math -ffp-contract=on %s -emit-llvm \
// RUN: -o - | FileCheck --check-prefixes CHECK,CHECK-ONFAST %s

// RUN: %clang_cc1 -triple=x86_64 -ffast-math -ffp-contract=fast %s -emit-llvm \
// RUN:  -o - | FileCheck --check-prefixes CHECK,CHECK-FASTFAST %s

// RUN: %clang_cc1 -triple=x86_64 -ffp-contract=fast -ffast-math  %s \
// RUN: -emit-llvm \
// RUN:  -o - | FileCheck --check-prefixes CHECK,CHECK-FASTFAST %s

// RUN: %clang_cc1 -triple=x86_64 -ffp-contract=off -fmath-errno \
// RUN: -fno-rounding-math %s -emit-llvm -o - \
// RUN:  -o - | FileCheck --check-prefixes CHECK,CHECK-NOFAST %s

// RUN: %clang -S -emit-llvm -fno-fast-math %s -o - \
// RUN: | FileCheck %s --check-prefixes=CHECK,CHECK-FPC-ON

// RUN: %clang -S -emit-llvm -ffp-contract=fast -fno-fast-math \
// RUN: %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-CONTRACTFAST

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

  // CHECK-DEFAULT: load float, ptr
  // CHECK-DEFAULT: fmul float
  // CHECK-DEFAULT: load float, ptr
  // CHECK-DEFAULT: fadd float

  // CHECK-ON: load float, ptr
  // CHECK-ON: load float, ptr
  // CHECK-ON: load float, ptr
  // CHECK-ON: call float @llvm.fmuladd.f32(float {{.*}}, float {{.*}}, float {{.*}})

  // CHECK-CONTRACTFAST: load float, ptr
  // CHECK-CONTRACTFAST: load float, ptr
  // CHECK-CONTRACTFAST: fmul contract float
  // CHECK-CONTRACTFAST: load float, ptr
  // CHECK-CONTRACTFAST: fadd contract float

  // CHECK-CONTRACTOFF: load float, ptr
  // CHECK-CONTRACTOFF: load float, ptr
  // CHECK-CONTRACTOFF: fmul reassoc nnan ninf nsz arcp afn float
  // CHECK-CONTRACTOFF: load float, ptr
  // CHECK-CONTRACTOFF: fadd reassoc nnan ninf nsz arcp afn float {{.*}}, {{.*}}

  // CHECK-ONFAST: load float, ptr
  // CHECK-ONFAST: load float, ptr
  // CHECK-ONFAST: load float, ptr
  // CHECK-ONFAST: call reassoc nnan ninf nsz arcp afn float @llvm.fmuladd.f32(float {{.*}}, float {{.*}}, float {{.*}})

  // CHECK-FASTFAST: load float, ptr
  // CHECK-FASTFAST: load float, ptr
  // CHECK-FASTFAST: fmul fast float
  // CHECK-FASTFAST: load float, ptr
  // CHECK-FASTFAST: fadd fast float {{.*}}, {{.*}}

  // CHECK-NOFAST: load float, ptr
  // CHECK-NOFAST: load float, ptr
  // CHECK-NOFAST: fmul float {{.*}}, {{.*}}
  // CHECK-NOFAST: load float, ptr
  // CHECK-NOFAST: fadd float {{.*}}, {{.*}}

  // CHECK-FPC-ON: load float, ptr
  // CHECK-FPC-ON: load float, ptr
  // CHECK-FPC-ON: load float, ptr
  // CHECK-FPC-ON: call float @llvm.fmuladd.f32(float {{.*}}, float {{.*}}, float {{.*}})

  // CHECK-FPC-OFF: load float, ptr
  // CHECK-FPC-OFF: load float, ptr
  // CHECK-FPC-OFF: fmul float
  // CHECK-FPC-OFF: load float, ptr
  // CHECK-FPC-OFF: fadd float {{.*}}, {{.*}}

  // CHECK-FFPC-OFF: load float, ptr
  // CHECK-FFPC-OFF: load float, ptr
  // CHECK-FPSC-OFF: call float @llvm.experimental.constrained.fmul.f32(float {{.*}}, float {{.*}}, {{.*}})
  // CHECK-FPSC-OFF: load float, ptr
  // CHECK-FPSC-OFF: [[RES:%.*]] = call float @llvm.experimental.constrained.fadd.f32(float {{.*}}, float {{.*}}, {{.*}})

}
