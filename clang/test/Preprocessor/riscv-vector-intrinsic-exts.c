// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32iv -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ALL-INTRINSICS %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64iv -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ALL-INTRINSICS %s

// Base vector intrinsics
// CHECK-ALL-INTRINSICS: __riscv_v_intrinsic_v 1

// Andes vendor extensions
// CHECK-ALL-INTRINSICS-NEXT: __riscv_v_intrinsic_xandesvbfhcvt 1
// CHECK-ALL-INTRINSICS-NEXT: __riscv_v_intrinsic_xandesvdot 1
// CHECK-ALL-INTRINSICS-NEXT: __riscv_v_intrinsic_xandesvpackfph 1
// CHECK-ALL-INTRINSICS-NEXT: __riscv_v_intrinsic_xandesvsintload 1

// SiFive vendor extensions
// CHECK-ALL-INTRINSICS-NEXT: __riscv_v_intrinsic_xsfmm32a16f 1
// CHECK-ALL-INTRINSICS-NEXT: __riscv_v_intrinsic_xsfmm32a32f 1
// CHECK-ALL-INTRINSICS-NEXT: __riscv_v_intrinsic_xsfmm32a8f 1
// CHECK-ALL-INTRINSICS-NEXT: __riscv_v_intrinsic_xsfmm32a8i 1
// CHECK-ALL-INTRINSICS-NEXT: __riscv_v_intrinsic_xsfmm64a64f 1
// CHECK-ALL-INTRINSICS-NEXT: __riscv_v_intrinsic_xsfmmbase 1
// CHECK-ALL-INTRINSICS-NEXT: __riscv_v_intrinsic_xsfvcp 1
// CHECK-ALL-INTRINSICS-NEXT: __riscv_v_intrinsic_xsfvfbfexp16e 1
// CHECK-ALL-INTRINSICS-NEXT: __riscv_v_intrinsic_xsfvfexp16e 1
// CHECK-ALL-INTRINSICS-NEXT: __riscv_v_intrinsic_xsfvfexp32e 1
// CHECK-ALL-INTRINSICS-NEXT: __riscv_v_intrinsic_xsfvfexpa 1
// CHECK-ALL-INTRINSICS-NEXT: __riscv_v_intrinsic_xsfvfexpa64e 1
// CHECK-ALL-INTRINSICS-NEXT: __riscv_v_intrinsic_xsfvfnrclipxfqf 1
// CHECK-ALL-INTRINSICS-NEXT: __riscv_v_intrinsic_xsfvfwmaccqqq 1
// CHECK-ALL-INTRINSICS-NEXT: __riscv_v_intrinsic_xsfvqmaccdod 1
// CHECK-ALL-INTRINSICS-NEXT: __riscv_v_intrinsic_xsfvqmaccqoq 1

// Standard vector extensions
// CHECK-ALL-INTRINSICS-NEXT: __riscv_v_intrinsic_zvabd 1
// CHECK-ALL-INTRINSICS-NEXT: __riscv_v_intrinsic_zvbb 1
// CHECK-ALL-INTRINSICS-NEXT: __riscv_v_intrinsic_zvbc 1
// CHECK-ALL-INTRINSICS-NEXT: __riscv_v_intrinsic_zvdot4a8i 1
// CHECK-ALL-INTRINSICS-NEXT: __riscv_v_intrinsic_zve32f 1
// CHECK-ALL-INTRINSICS-NEXT: __riscv_v_intrinsic_zve32x 1
// CHECK-ALL-INTRINSICS-NEXT: __riscv_v_intrinsic_zve64d 1
// CHECK-ALL-INTRINSICS-NEXT: __riscv_v_intrinsic_zve64f 1
// CHECK-ALL-INTRINSICS-NEXT: __riscv_v_intrinsic_zve64x 1
// CHECK-ALL-INTRINSICS-NEXT: __riscv_v_intrinsic_zvfbfa 1
// CHECK-ALL-INTRINSICS-NEXT: __riscv_v_intrinsic_zvfbfmin 1
// CHECK-ALL-INTRINSICS-NEXT: __riscv_v_intrinsic_zvfbfwma 1
// CHECK-ALL-INTRINSICS-NEXT: __riscv_v_intrinsic_zvfh 1
// CHECK-ALL-INTRINSICS-NEXT: __riscv_v_intrinsic_zvfhmin 1
// CHECK-ALL-INTRINSICS-NEXT: __riscv_v_intrinsic_zvfofp8min 1
// CHECK-ALL-INTRINSICS-NEXT: __riscv_v_intrinsic_zvkb 1
// CHECK-ALL-INTRINSICS-NEXT: __riscv_v_intrinsic_zvkg 1
// CHECK-ALL-INTRINSICS-NEXT: __riscv_v_intrinsic_zvkn 1
// CHECK-ALL-INTRINSICS-NEXT: __riscv_v_intrinsic_zvknc 1
// CHECK-ALL-INTRINSICS-NEXT: __riscv_v_intrinsic_zvkned 1
// CHECK-ALL-INTRINSICS-NEXT: __riscv_v_intrinsic_zvkng 1
// CHECK-ALL-INTRINSICS-NEXT: __riscv_v_intrinsic_zvknha 1
// CHECK-ALL-INTRINSICS-NEXT: __riscv_v_intrinsic_zvknhb 1
// CHECK-ALL-INTRINSICS-NEXT: __riscv_v_intrinsic_zvks 1
// CHECK-ALL-INTRINSICS-NEXT: __riscv_v_intrinsic_zvksc 1
// CHECK-ALL-INTRINSICS-NEXT: __riscv_v_intrinsic_zvksed 1
// CHECK-ALL-INTRINSICS-NEXT: __riscv_v_intrinsic_zvksg 1
// CHECK-ALL-INTRINSICS-NEXT: __riscv_v_intrinsic_zvksh 1
