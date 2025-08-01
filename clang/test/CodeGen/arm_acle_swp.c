// RUN: %clang_cc1 -ffreestanding -triple thumbv7m-none-eabi -O0 -disable-O0-optnone -emit-llvm -o - %s | opt -S -passes=mem2reg | FileCheck %s -check-prefix=ATOMIC
// RUN: %clang_cc1 -ffreestanding -triple armv7a-none-eabi -O0 -disable-O0-optnone -emit-llvm -o - %s | opt -S -passes=mem2reg | FileCheck %s -check-prefix=ATOMIC
// RUN: %clang_cc1 -ffreestanding -triple armv6-none-eabi -O0 -disable-O0-optnone -emit-llvm -o - %s | opt -S -passes=mem2reg | FileCheck %s -check-prefix=ATOMIC
// RUN: %clang_cc1 -ffreestanding -triple thumbv6m-none-eabi -O0 -disable-O0-optnone -emit-llvm -o - %s | opt -S -passes=mem2reg | FileCheck %s -check-prefix=ATOMIC
// RUN: %clang_cc1 -ffreestanding -triple armv5-unknown-linux-gnu -target-abi aapcs -O0 -disable-O0-optnone -emit-llvm -o - %s | opt -S -passes=mem2reg | FileCheck %s -check-prefix=ATOMIC
// RUN: %clang_cc1 -ffreestanding -triple armv5-none-eabi -O0 -disable-O0-optnone -emit-llvm -o - %s | opt -S -passes=mem2reg | FileCheck %s -check-prefix=SWP

// REQUIRES: arm-registered-target

#include <arm_acle.h>

// SWP:    call i32 asm "swp $0, $1, [$2]", "=r,r,r,~{memory}"(i32 {{.*}}, ptr {{.*}})

// ATOMIC: atomicrmw volatile xchg ptr {{.*}}, i32 {{.*}} monotonic, align 4
uint32_t test_swp(uint32_t x, volatile void *p) {
  return __swp(x, p);
}
