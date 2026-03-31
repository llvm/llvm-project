// Tests for RISC-V intrinsic detection macros.
// These macros indicate which extensions have intrinsics supported by the
// toolchain, regardless of whether they are currently enabled via -march.

// RUN: %clang_cc1 -triple riscv32 -E -dM %s -o - \
// RUN:   | FileCheck --check-prefix=CHECK-SCALAR-EXTS %s
// RUN: %clang_cc1 -triple riscv64 -E -dM %s -o - \
// RUN:   | FileCheck --check-prefix=CHECK-SCALAR-EXTS %s

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32iv -E -dM %s -o - \
// RUN:   | FileCheck --check-prefix=CHECK-VECTOR-EXTS %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64iv -E -dM %s -o - \
// RUN:   | FileCheck --check-prefix=CHECK-VECTOR-EXTS %s

// Scalar intrinsic extension macros (__riscv_intrinsic_*)
// CHECK-SCALAR-EXTS-DAG: #define __riscv_intrinsic_zbb 1
// CHECK-SCALAR-EXTS-DAG: #define __riscv_intrinsic_zbc 1
// CHECK-SCALAR-EXTS-DAG: #define __riscv_intrinsic_zbkb 1
// CHECK-SCALAR-EXTS-DAG: #define __riscv_intrinsic_zbkc 1
// CHECK-SCALAR-EXTS-DAG: #define __riscv_intrinsic_zbkx 1
// CHECK-SCALAR-EXTS-DAG: #define __riscv_intrinsic_zknd 1
// CHECK-SCALAR-EXTS-DAG: #define __riscv_intrinsic_zkne 1
// CHECK-SCALAR-EXTS-DAG: #define __riscv_intrinsic_zknh 1
// CHECK-SCALAR-EXTS-DAG: #define __riscv_intrinsic_zksed 1
// CHECK-SCALAR-EXTS-DAG: #define __riscv_intrinsic_zksh 1
// CHECK-SCALAR-EXTS-DAG: #define __riscv_intrinsic_xtheadbb 1

// Scalar composite extensions (defined when all components are supported)
// CHECK-SCALAR-EXTS-DAG: #define __riscv_intrinsic_zkn 1
// CHECK-SCALAR-EXTS-DAG: #define __riscv_intrinsic_zks 1

// Vector intrinsic extension macros (__riscv_v_intrinsic_*)

// Base vector intrinsics
// CHECK-VECTOR-EXTS: __riscv_v_intrinsic_v 1

// Andes vendor extensions
// CHECK-VECTOR-EXTS-NEXT: __riscv_v_intrinsic_xandesvbfhcvt 1
// CHECK-VECTOR-EXTS-NEXT: __riscv_v_intrinsic_xandesvdot 1
// CHECK-VECTOR-EXTS-NEXT: __riscv_v_intrinsic_xandesvpackfph 1
// CHECK-VECTOR-EXTS-NEXT: __riscv_v_intrinsic_xandesvsintload 1

// SiFive vendor extensions
// CHECK-VECTOR-EXTS-NEXT: __riscv_v_intrinsic_xsfmm32a16f 1
// CHECK-VECTOR-EXTS-NEXT: __riscv_v_intrinsic_xsfmm32a32f 1
// CHECK-VECTOR-EXTS-NEXT: __riscv_v_intrinsic_xsfmm32a8f 1
// CHECK-VECTOR-EXTS-NEXT: __riscv_v_intrinsic_xsfmm32a8i 1
// CHECK-VECTOR-EXTS-NEXT: __riscv_v_intrinsic_xsfmm64a64f 1
// CHECK-VECTOR-EXTS-NEXT: __riscv_v_intrinsic_xsfmmbase 1
// CHECK-VECTOR-EXTS-NEXT: __riscv_v_intrinsic_xsfvcp 1
// CHECK-VECTOR-EXTS-NEXT: __riscv_v_intrinsic_xsfvfbfexp16e 1
// CHECK-VECTOR-EXTS-NEXT: __riscv_v_intrinsic_xsfvfexp16e 1
// CHECK-VECTOR-EXTS-NEXT: __riscv_v_intrinsic_xsfvfexp32e 1
// CHECK-VECTOR-EXTS-NEXT: __riscv_v_intrinsic_xsfvfexpa 1
// CHECK-VECTOR-EXTS-NEXT: __riscv_v_intrinsic_xsfvfexpa64e 1
// CHECK-VECTOR-EXTS-NEXT: __riscv_v_intrinsic_xsfvfnrclipxfqf 1
// CHECK-VECTOR-EXTS-NEXT: __riscv_v_intrinsic_xsfvfwmaccqqq 1
// CHECK-VECTOR-EXTS-NEXT: __riscv_v_intrinsic_xsfvqmaccdod 1
// CHECK-VECTOR-EXTS-NEXT: __riscv_v_intrinsic_xsfvqmaccqoq 1

// Standard vector extensions
// CHECK-VECTOR-EXTS-NEXT: __riscv_v_intrinsic_zvabd 1
// CHECK-VECTOR-EXTS-NEXT: __riscv_v_intrinsic_zvbb 1
// CHECK-VECTOR-EXTS-NEXT: __riscv_v_intrinsic_zvbc 1
// CHECK-VECTOR-EXTS-NEXT: __riscv_v_intrinsic_zvdot4a8i 1
// CHECK-VECTOR-EXTS-NEXT: __riscv_v_intrinsic_zve32f 1
// CHECK-VECTOR-EXTS-NEXT: __riscv_v_intrinsic_zve32x 1
// CHECK-VECTOR-EXTS-NEXT: __riscv_v_intrinsic_zve64d 1
// CHECK-VECTOR-EXTS-NEXT: __riscv_v_intrinsic_zve64f 1
// CHECK-VECTOR-EXTS-NEXT: __riscv_v_intrinsic_zve64x 1
// CHECK-VECTOR-EXTS-NEXT: __riscv_v_intrinsic_zvfbfa 1
// CHECK-VECTOR-EXTS-NEXT: __riscv_v_intrinsic_zvfbfmin 1
// CHECK-VECTOR-EXTS-NEXT: __riscv_v_intrinsic_zvfbfwma 1
// CHECK-VECTOR-EXTS-NEXT: __riscv_v_intrinsic_zvfh 1
// CHECK-VECTOR-EXTS-NEXT: __riscv_v_intrinsic_zvfhmin 1
// CHECK-VECTOR-EXTS-NEXT: __riscv_v_intrinsic_zvfofp8min 1
// CHECK-VECTOR-EXTS-NEXT: __riscv_v_intrinsic_zvkb 1
// CHECK-VECTOR-EXTS-NEXT: __riscv_v_intrinsic_zvkg 1
// CHECK-VECTOR-EXTS-NEXT: __riscv_v_intrinsic_zvkn 1
// CHECK-VECTOR-EXTS-NEXT: __riscv_v_intrinsic_zvknc 1
// CHECK-VECTOR-EXTS-NEXT: __riscv_v_intrinsic_zvkned 1
// CHECK-VECTOR-EXTS-NEXT: __riscv_v_intrinsic_zvkng 1
// CHECK-VECTOR-EXTS-NEXT: __riscv_v_intrinsic_zvknha 1
// CHECK-VECTOR-EXTS-NEXT: __riscv_v_intrinsic_zvknhb 1
// CHECK-VECTOR-EXTS-NEXT: __riscv_v_intrinsic_zvks 1
// CHECK-VECTOR-EXTS-NEXT: __riscv_v_intrinsic_zvksc 1
// CHECK-VECTOR-EXTS-NEXT: __riscv_v_intrinsic_zvksed 1
// CHECK-VECTOR-EXTS-NEXT: __riscv_v_intrinsic_zvksg 1
// CHECK-VECTOR-EXTS-NEXT: __riscv_v_intrinsic_zvksh 1

