// Tests for RISC-V intrinsic detection macros.
// These macros indicate which extensions have intrinsics supported by the
// toolchain, regardless of whether they are currently enabled via -march.

// REQUIRES: riscv-registered-target

// RUN: %clang_cc1 -triple riscv32 -target-feature +zihintntl -E -dM %s -o - \
// RUN:   | FileCheck --check-prefix=CHECK-INTRINSIC-EXTS %s
// RUN: %clang_cc1 -triple riscv64 -target-feature +zihintntl -E -dM %s -o - \
// RUN:   | FileCheck --check-prefix=CHECK-INTRINSIC-EXTS %s

#include <riscv_bitmanip.h>
#include <riscv_corev_alu.h>
#include <riscv_crypto.h>
#include <riscv_mips.h>
#include <riscv_nds.h>
#include <riscv_ntlh.h>
#include <riscv_vector.h>
#include <andes_vector.h>
#include <sifive_vector.h>

// CHECK-INTRINSIC-EXTS: #define __riscv_intrinsic_v 1
// CHECK-INTRINSIC-EXTS-NEXT: #define __riscv_intrinsic_xandesbfhcvt 1
// CHECK-INTRINSIC-EXTS-NEXT: #define __riscv_intrinsic_xandesperf 1
// CHECK-INTRINSIC-EXTS-NEXT: #define __riscv_intrinsic_xandesvbfhcvt 1
// CHECK-INTRINSIC-EXTS-NEXT: #define __riscv_intrinsic_xandesvdot 1
// CHECK-INTRINSIC-EXTS-NEXT: #define __riscv_intrinsic_xandesvpackfph 1
// CHECK-INTRINSIC-EXTS-NEXT: #define __riscv_intrinsic_xandesvsintload 1
// CHECK-INTRINSIC-EXTS-NEXT: #define __riscv_intrinsic_xcvalu 1
// CHECK-INTRINSIC-EXTS-NEXT: #define __riscv_intrinsic_xmipsexectl 1
// CHECK-INTRINSIC-EXTS-NEXT: #define __riscv_intrinsic_xsfmm32a16f 1
// CHECK-INTRINSIC-EXTS-NEXT: #define __riscv_intrinsic_xsfmm32a32f 1
// CHECK-INTRINSIC-EXTS-NEXT: #define __riscv_intrinsic_xsfmm32a8f 1
// CHECK-INTRINSIC-EXTS-NEXT: #define __riscv_intrinsic_xsfmm32a8i 1
// CHECK-INTRINSIC-EXTS-NEXT: #define __riscv_intrinsic_xsfmm64a64f 1
// CHECK-INTRINSIC-EXTS-NEXT: #define __riscv_intrinsic_xsfmmbase 1
// CHECK-INTRINSIC-EXTS-NEXT: #define __riscv_intrinsic_xsfvcp 1
// CHECK-INTRINSIC-EXTS-NEXT: #define __riscv_intrinsic_xsfvfbfexp16e 1
// CHECK-INTRINSIC-EXTS-NEXT: #define __riscv_intrinsic_xsfvfexp16e 1
// CHECK-INTRINSIC-EXTS-NEXT: #define __riscv_intrinsic_xsfvfexp32e 1
// CHECK-INTRINSIC-EXTS-NEXT: #define __riscv_intrinsic_xsfvfexpa 1
// CHECK-INTRINSIC-EXTS-NEXT: #define __riscv_intrinsic_xsfvfexpa64e 1
// CHECK-INTRINSIC-EXTS-NEXT: #define __riscv_intrinsic_xsfvfnrclipxfqf 1
// CHECK-INTRINSIC-EXTS-NEXT: #define __riscv_intrinsic_xsfvfwmaccqqq 1
// CHECK-INTRINSIC-EXTS-NEXT: #define __riscv_intrinsic_xsfvqmaccdod 1
// CHECK-INTRINSIC-EXTS-NEXT: #define __riscv_intrinsic_xsfvqmaccqoq 1
// CHECK-INTRINSIC-EXTS-NEXT: #define __riscv_intrinsic_zbb 1
// CHECK-INTRINSIC-EXTS-NEXT: #define __riscv_intrinsic_zbc 1
// CHECK-INTRINSIC-EXTS-NEXT: #define __riscv_intrinsic_zbkb 1
// CHECK-INTRINSIC-EXTS-NEXT: #define __riscv_intrinsic_zbkc 1
// CHECK-INTRINSIC-EXTS-NEXT: #define __riscv_intrinsic_zbkx 1
// CHECK-INTRINSIC-EXTS-NEXT: #define __riscv_intrinsic_zihintntl 1
// CHECK-INTRINSIC-EXTS-NEXT: #define __riscv_intrinsic_zkn 1
// CHECK-INTRINSIC-EXTS-NEXT: #define __riscv_intrinsic_zknd 1
// CHECK-INTRINSIC-EXTS-NEXT: #define __riscv_intrinsic_zkne 1
// CHECK-INTRINSIC-EXTS-NEXT: #define __riscv_intrinsic_zknh 1
// CHECK-INTRINSIC-EXTS-NEXT: #define __riscv_intrinsic_zks 1
// CHECK-INTRINSIC-EXTS-NEXT: #define __riscv_intrinsic_zksed 1
// CHECK-INTRINSIC-EXTS-NEXT: #define __riscv_intrinsic_zksh 1
// CHECK-INTRINSIC-EXTS-NEXT: #define __riscv_intrinsic_zvabd 1
// CHECK-INTRINSIC-EXTS-NEXT: #define __riscv_intrinsic_zvbb 1
// CHECK-INTRINSIC-EXTS-NEXT: #define __riscv_intrinsic_zvbc 1
// CHECK-INTRINSIC-EXTS-NEXT: #define __riscv_intrinsic_zvdot4a8i 1
// CHECK-INTRINSIC-EXTS-NEXT: #define __riscv_intrinsic_zve32f 1
// CHECK-INTRINSIC-EXTS-NEXT: #define __riscv_intrinsic_zve32x 1
// CHECK-INTRINSIC-EXTS-NEXT: #define __riscv_intrinsic_zve64d 1
// CHECK-INTRINSIC-EXTS-NEXT: #define __riscv_intrinsic_zve64f 1
// CHECK-INTRINSIC-EXTS-NEXT: #define __riscv_intrinsic_zve64x 1
// CHECK-INTRINSIC-EXTS-NEXT: #define __riscv_intrinsic_zvfbfa 1
// CHECK-INTRINSIC-EXTS-NEXT: #define __riscv_intrinsic_zvfbfmin 1
// CHECK-INTRINSIC-EXTS-NEXT: #define __riscv_intrinsic_zvfbfwma 1
// CHECK-INTRINSIC-EXTS-NEXT: #define __riscv_intrinsic_zvfh 1
// CHECK-INTRINSIC-EXTS-NEXT: #define __riscv_intrinsic_zvfhmin 1
// CHECK-INTRINSIC-EXTS-NEXT: #define __riscv_intrinsic_zvfofp8min 1
// CHECK-INTRINSIC-EXTS-NEXT: #define __riscv_intrinsic_zvkb 1
// CHECK-INTRINSIC-EXTS-NEXT: #define __riscv_intrinsic_zvkg 1
// CHECK-INTRINSIC-EXTS-NEXT: #define __riscv_intrinsic_zvkn 1
// CHECK-INTRINSIC-EXTS-NEXT: #define __riscv_intrinsic_zvknc 1
// CHECK-INTRINSIC-EXTS-NEXT: #define __riscv_intrinsic_zvkned 1
// CHECK-INTRINSIC-EXTS-NEXT: #define __riscv_intrinsic_zvkng 1
// CHECK-INTRINSIC-EXTS-NEXT: #define __riscv_intrinsic_zvknha 1
// CHECK-INTRINSIC-EXTS-NEXT: #define __riscv_intrinsic_zvknhb 1
// CHECK-INTRINSIC-EXTS-NEXT: #define __riscv_intrinsic_zvks 1
// CHECK-INTRINSIC-EXTS-NEXT: #define __riscv_intrinsic_zvksc 1
// CHECK-INTRINSIC-EXTS-NEXT: #define __riscv_intrinsic_zvksed 1
// CHECK-INTRINSIC-EXTS-NEXT: #define __riscv_intrinsic_zvksg 1
// CHECK-INTRINSIC-EXTS-NEXT: #define __riscv_intrinsic_zvksh 1
