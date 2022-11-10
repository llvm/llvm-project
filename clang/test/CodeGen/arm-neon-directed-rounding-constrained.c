// RUN: %clang_cc1 -triple thumbv8-linux-gnueabihf -target-cpu cortex-a57 \
// RUN:     -ffreestanding -disable-O0-optnone -emit-llvm %s -o - | \
// RUN:     opt -S -passes=mem2reg | FileCheck -check-prefixes=COMMON,COMMONIR,UNCONSTRAINED %s
// RUN: %clang_cc1 -triple arm64-linux-gnueabihf -target-feature +neon \
// RUN:     -ffreestanding -disable-O0-optnone -emit-llvm %s -o - | \
// RUN:     opt -S -passes=mem2reg | FileCheck -check-prefixes=COMMON,COMMONIR,UNCONSTRAINED %s

// RUN: %clang_cc1 -triple thumbv8-linux-gnueabihf -target-cpu cortex-a57 \
// RUN:     -ffp-exception-behavior=strict \
// RUN:     -fexperimental-strict-floating-point \
// RUN:     -ffreestanding -disable-O0-optnone -emit-llvm %s -o - | \
// RUN:     opt -S -passes=mem2reg | FileCheck -check-prefixes=COMMON,COMMONIR,CONSTRAINED %s
// RUN: %clang_cc1 -triple arm64-linux-gnueabihf -target-feature +neon \
// RUN:     -ffp-exception-behavior=strict \
// RUN:     -fexperimental-strict-floating-point \
// RUN:     -ffreestanding -disable-O0-optnone -emit-llvm %s -o - | \
// RUN:     opt -S -passes=mem2reg | FileCheck -check-prefixes=COMMON,COMMONIR,CONSTRAINED %s

// RUN: %clang_cc1 -triple thumbv8-linux-gnueabihf -target-cpu cortex-a57 \
// RUN:     -ffreestanding -disable-O0-optnone -emit-llvm %s -o - | \
// RUN:     opt -S -passes=mem2reg | llc -o=- - | FileCheck -check-prefixes=COMMON,CHECK-ASM32 %s
// RUN: %clang_cc1 -triple arm64-linux-gnueabihf -target-feature +neon \
// RUN:     -ffreestanding -disable-O0-optnone -emit-llvm %s -o - | \
// RUN:     opt -S -passes=mem2reg | llc -o=- - | FileCheck -check-prefixes=COMMON,CHECK-ASM64 %s

// RUN: %clang_cc1 -triple thumbv8-linux-gnueabihf -target-cpu cortex-a57 \
// RUN:     -ffp-exception-behavior=strict \
// RUN:     -fexperimental-strict-floating-point \
// RUN:     -ffreestanding -disable-O0-optnone -emit-llvm %s -o - | \
// RUN:     opt -S -passes=mem2reg | llc -o=- - | FileCheck -check-prefixes=COMMON,CHECK-ASM32 %s
// RUN: %clang_cc1 -triple arm64-linux-gnueabihf -target-feature +neon \
// RUN:     -ffp-exception-behavior=strict \
// RUN:     -fexperimental-strict-floating-point \
// RUN:     -ffreestanding -disable-O0-optnone -emit-llvm %s -o - | \
// RUN:     opt -S -passes=mem2reg | llc -o=- - | FileCheck -check-prefixes=COMMON,CHECK-ASM64 %s

// REQUIRES: arm-registered-target,aarch64-registered-target

#include <arm_neon.h>

// COMMON-LABEL: test_vrndi_f32
// COMMONIR:      [[TMP0:%.*]] = bitcast <2 x float> %a to <8 x i8>
// UNCONSTRAINED: [[VRNDI1_I:%.*]] = call <2 x float> @llvm.nearbyint.v2f32(<2 x float> %a)
// CONSTRAINED:   [[VRNDI1_I:%.*]] = call <2 x float> @llvm.experimental.constrained.nearbyint.v2f32(<2 x float> %a, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK-ASM32:   vrintr.f32 s{{[0-9]+}}, s{{[0-9]+}}
// CHECK-ASM32:   vrintr.f32 s{{[0-9]+}}, s{{[0-9]+}}
// CHECK-ASM64:   frinti v{{[0-9]+}}.2s, v{{[0-9]+}}.2s
// COMMONIR:      ret <2 x float> [[VRNDI1_I]]
float32x2_t test_vrndi_f32(float32x2_t a) {
  return vrndi_f32(a);
}

// COMMON-LABEL: test_vrndiq_f32
// COMMONIR:      [[TMP0:%.*]] = bitcast <4 x float> %a to <16 x i8>
// UNCONSTRAINED: [[VRNDI1_I:%.*]] = call <4 x float> @llvm.nearbyint.v4f32(<4 x float> %a)
// CONSTRAINED:   [[VRNDI1_I:%.*]] = call <4 x float> @llvm.experimental.constrained.nearbyint.v4f32(<4 x float> %a, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK-ASM32:   vrintr.f32 s{{[0-9]+}}, s{{[0-9]+}}
// CHECK-ASM32:   vrintr.f32 s{{[0-9]+}}, s{{[0-9]+}}
// CHECK-ASM32:   vrintr.f32 s{{[0-9]+}}, s{{[0-9]+}}
// CHECK-ASM32:   vrintr.f32 s{{[0-9]+}}, s{{[0-9]+}}
// CHECK-ASM64:   frinti v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
// COMMONIR:      ret <4 x float> [[VRNDI1_I]]
float32x4_t test_vrndiq_f32(float32x4_t a) {
  return vrndiq_f32(a);
}
