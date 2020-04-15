// RUN: %clang -target riscv32-unknown-linux-gnu -march=rv32i -x c -E -dM %s \
// RUN: -o - | FileCheck %s
// RUN: %clang -target riscv64-unknown-linux-gnu -march=rv64i -x c -E -dM %s \
// RUN: -o - | FileCheck %s

// CHECK-NOT: __riscv_div
// CHECK-NOT: __riscv_mul
// CHECK-NOT: __riscv_muldiv
// CHECK-NOT: __riscv_compressed
// CHECK-NOT: __riscv_bitmanip
// CHECK-NOT: __riscv_flen
// CHECK-NOT: __riscv_fdiv
// CHECK-NOT: __riscv_fsqrt
// CHECK-NOT: __riscv_atomic

// RUN: %clang -target riscv32-unknown-linux-gnu -march=rv32im -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-M-EXT %s
// RUN: %clang -target riscv64-unknown-linux-gnu -march=rv64im -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-M-EXT %s
// CHECK-M-EXT: __riscv_div 1
// CHECK-M-EXT: __riscv_mul 1
// CHECK-M-EXT: __riscv_muldiv 1

// RUN: %clang -target riscv32-unknown-linux-gnu -march=rv32ia -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-A-EXT %s
// RUN: %clang -target riscv64-unknown-linux-gnu -march=rv64ia -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-A-EXT %s
// CHECK-A-EXT: __riscv_atomic 1

// RUN: %clang -target riscv32-unknown-linux-gnu -march=rv32if -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-F-EXT %s
// RUN: %clang -target riscv64-unknown-linux-gnu -march=rv64if -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-F-EXT %s
// CHECK-F-EXT: __riscv_fdiv 1
// CHECK-F-EXT: __riscv_flen 32
// CHECK-F-EXT: __riscv_fsqrt 1

// RUN: %clang -target riscv32-unknown-linux-gnu -march=rv32ifd -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-D-EXT %s
// RUN: %clang -target riscv64-unknown-linux-gnu -march=rv64ifd -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-D-EXT %s
// CHECK-D-EXT: __riscv_fdiv 1
// CHECK-D-EXT: __riscv_flen 64
// CHECK-D-EXT: __riscv_fsqrt 1

// RUN: %clang -target riscv32-unknown-linux-gnu -march=rv32ic -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-C-EXT %s
// RUN: %clang -target riscv64-unknown-linux-gnu -march=rv64ic -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-C-EXT %s
// CHECK-C-EXT: __riscv_compressed 1

// RUN: %clang -target riscv32-unknown-linux-gnu -menable-experimental-extensions -march=rv32ib0p92 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-B-EXT %s
// RUN: %clang -target riscv64-unknown-linux-gnu -menable-experimental-extensions -march=rv64ib0p92 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-B-EXT %s
// CHECK-B-EXT: __riscv_bitmanip 1

// RUN: %clang -target riscv32-unknown-linux-gnu -march=rv32ifd -mabi=ilp32 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-SOFT %s
// RUN: %clang -target riscv64-unknown-linux-gnu -march=rv64ifd -mabi=lp64 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-SOFT %s
// CHECK-SOFT: __riscv_float_abi_soft 1
// CHECK-SOFT-NOT: __riscv_float_abi_single
// CHECK-SOFT-NOT: __riscv_float_abi_double

// RUN: %clang -target riscv32-unknown-linux-gnu -march=rv32ifd -mabi=ilp32f -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-SINGLE %s
// RUN: %clang -target riscv64-unknown-linux-gnu -march=rv64ifd -mabi=lp64f -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-SINGLE %s
// CHECK-SINGLE: __riscv_float_abi_single 1
// CHECK-SINGLE-NOT: __riscv_float_abi_soft
// CHECK-SINGLE-NOT: __riscv_float_abi_double

// RUN: %clang -target riscv32-unknown-linux-gnu -march=rv32ifd -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-DOUBLE %s
// RUN: %clang -target riscv64-unknown-linux-gnu -march=rv64ifd -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-DOUBLE %s
// CHECK-DOUBLE: __riscv_float_abi_double 1
// CHECK-DOUBLE-NOT: __riscv_float_abi_soft
// CHECK-DOUBLE-NOT: __riscv_float_abi_single
