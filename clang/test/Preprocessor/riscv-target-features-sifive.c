// RUN: %clang --target=riscv32 \
// RUN:   -march=rv32i_zve32x_xsfmm128t -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XSFMM128T %s
// RUN: %clang --target=riscv64 \
// RUN:   -march=rv64i_zve32x_xsfmm128t -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XSFMM128T %s
// CHECK-XSFMM128T: __riscv_xsfmm128t  6000{{$}}
//
// RUN: %clang --target=riscv32 \
// RUN:   -march=rv32i_zve32x_xsfmm16t -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XSFMM16T %s
// RUN: %clang --target=riscv64 \
// RUN:   -march=rv64i_zve32x_xsfmm16t -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XSFMM16T %s
// CHECK-XSFMM16T: __riscv_xsfmm16t  6000{{$}}

// RUN: %clang --target=riscv32 \
// RUN:   -march=rv32i_zve32x_xsfmm32a8i -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XSFMM32a8I %s
// RUN: %clang --target=riscv64 \
// RUN:   -march=rv64i_zve32x_xsfmm32a8i -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XSFMM32a8I %s
// CHECK-XSFMM32a8I: __riscv_xsfmm32a8i  6000{{$}}

// RUN: %clang --target=riscv32 \
// RUN:   -march=rv32i_zve32x_xsfmm32a8f -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XSFMM32A8F %s
// RUN: %clang --target=riscv64 \
// RUN:   -march=rv64i_zve32x_xsfmm32a8f -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XSFMM32A8F %s
// CHECK-XSFMM32A8F: __riscv_xsfmm32a8f  6000{{$}}

// RUN: %clang --target=riscv32 \
// RUN:   -march=rv32i_zve32x_xsfmm32a16f -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XSFMM32a16F %s
// RUN: %clang --target=riscv64 \
// RUN:   -march=rv64i_zve32x_xsfmm32a16f -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XSFMM32a16F %s
// CHECK-XSFMM32a16F: __riscv_xsfmm32a16f  6000{{$}}

// RUN: %clang --target=riscv32 \
// RUN:   -march=rv32i_zve32x_xsfmm32a32f -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XSFMM32a32F %s
// RUN: %clang --target=riscv64 \
// RUN:   -march=rv64i_zve32x_xsfmm32a32f -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XSFMM32a32F %s
// CHECK-XSFMM32a32F: __riscv_xsfmm32a32f  6000{{$}}

// RUN: %clang --target=riscv32 \
// RUN:   -march=rv32i_zve32x_xsfmm32t -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XSFMM32T %s
// RUN: %clang --target=riscv64 \
// RUN:   -march=rv64i_zve32x_xsfmm32t -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XSFMM32T %s
// CHECK-XSFMM32T: __riscv_xsfmm32t  6000{{$}}

// RUN: %clang --target=riscv32 \
// RUN:   -march=rv32i_zve32x_xsfmm64a64f -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XSFMM64a64f %s
// RUN: %clang --target=riscv64 \
// RUN:   -march=rv64i_zve32x_xsfmm64a64f -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XSFMM64a64f %s
// CHECK-XSFMM64a64f: __riscv_xsfmm64a64f  6000{{$}}

// RUN: %clang --target=riscv32 \
// RUN:   -march=rv32i_zve32x_xsfmm64t -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XSFMM64T %s
// RUN: %clang --target=riscv64 \
// RUN:   -march=rv64i_zve32x_xsfmm64t -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XSFMM64T %s
// CHECK-XSFMM64T: __riscv_xsfmm64t  6000{{$}}

// RUN: %clang --target=riscv32 \
// RUN:   -march=rv32i_zve32x_xsfmmbase -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XSFMMBASE %s
// RUN: %clang --target=riscv64 \
// RUN:   -march=rv64i_zve32x_xsfmmbase -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XSFMMBASE %s
// CHECK-XSFMMBASE: __riscv_xsfmmbase  6000{{$}}
