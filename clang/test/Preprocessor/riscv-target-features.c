// RUN: %clang -target riscv32-unknown-linux-gnu -march=rv32i -x c -E -dM %s \
// RUN: -o - | FileCheck %s
// RUN: %clang -target riscv64-unknown-linux-gnu -march=rv64i -x c -E -dM %s \
// RUN: -o - | FileCheck %s

// CHECK-NOT: __riscv_div
// CHECK-NOT: __riscv_m
// CHECK-NOT: __riscv_mul
// CHECK-NOT: __riscv_muldiv
// CHECK-NOT: __riscv_a 2000000
// CHECK-NOT: __riscv_atomic
// CHECK-NOT: __riscv_f 2000000
// CHECK-NOT: __riscv_d
// CHECK-NOT: __riscv_flen
// CHECK-NOT: __riscv_fdiv
// CHECK-NOT: __riscv_fsqrt
// CHECK-NOT: __riscv_c 2000000
// CHECK-NOT: __riscv_compressed
// CHECK-NOT: __riscv_b
// CHECK-NOT: __riscv_bitmanip
// CHECK-NOT: __riscv_zba
// CHECK-NOT: __riscv_zbb
// CHECK-NOT: __riscv_zbc
// CHECK-NOT: __riscv_zbe
// CHECK-NOT: __riscv_zbf
// CHECK-NOT: __riscv_zbm
// CHECK-NOT: __riscv_zbp
// CHECK-NOT: __riscv_zbr
// CHECK-NOT: __riscv_zbs
// CHECK-NOT: __riscv_zbt
// CHECK-NOT: __riscv_zfh
// CHECK-NOT: __riscv_v
// CHECK-NOT: __riscv_vector
// CHECK-NOT: __riscv_zvamo
// CHECK-NOT: __riscv_zvlsseg

// RUN: %clang -target riscv32-unknown-linux-gnu -march=rv32im -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-M-EXT %s
// RUN: %clang -target riscv64-unknown-linux-gnu -march=rv64im -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-M-EXT %s
// CHECK-M-EXT: __riscv_div 1
// CHECK-M-EXT: __riscv_m 2000000
// CHECK-M-EXT: __riscv_mul 1
// CHECK-M-EXT: __riscv_muldiv 1

// RUN: %clang -target riscv32-unknown-linux-gnu -march=rv32ia -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-A-EXT %s
// RUN: %clang -target riscv64-unknown-linux-gnu -march=rv64ia -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-A-EXT %s
// CHECK-A-EXT: __riscv_a 2000000
// CHECK-A-EXT: __riscv_atomic 1

// RUN: %clang -target riscv32-unknown-linux-gnu -march=rv32if -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-F-EXT %s
// RUN: %clang -target riscv64-unknown-linux-gnu -march=rv64if -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-F-EXT %s
// CHECK-F-EXT: __riscv_f 2000000
// CHECK-F-EXT: __riscv_fdiv 1
// CHECK-F-EXT: __riscv_flen 32
// CHECK-F-EXT: __riscv_fsqrt 1

// RUN: %clang -target riscv32-unknown-linux-gnu -march=rv32ifd -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-D-EXT %s
// RUN: %clang -target riscv64-unknown-linux-gnu -march=rv64ifd -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-D-EXT %s
// CHECK-D-EXT: __riscv_d 2000000
// CHECK-D-EXT: __riscv_fdiv 1
// CHECK-D-EXT: __riscv_flen 64
// CHECK-D-EXT: __riscv_fsqrt 1

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

// RUN: %clang -target riscv32-unknown-linux-gnu -march=rv32ic -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-C-EXT %s
// RUN: %clang -target riscv64-unknown-linux-gnu -march=rv64ic -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-C-EXT %s
// CHECK-C-EXT: __riscv_c 2000000
// CHECK-C-EXT: __riscv_compressed 1

// RUN: %clang -target riscv32-unknown-linux-gnu -menable-experimental-extensions \
// RUN: -march=rv32ib0p93 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-B-EXT %s
// RUN: %clang -target riscv64-unknown-linux-gnu -menable-experimental-extensions \
// RUN: -march=rv64ib0p93 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-B-EXT %s
// CHECK-B-EXT: __riscv_b 93000
// CHECK-B-EXT: __riscv_bitmanip 1
// CHECK-B-EXT: __riscv_zba 93000
// CHECK-B-EXT: __riscv_zbb 93000
// CHECK-B-EXT: __riscv_zbc 93000
// CHECK-B-EXT: __riscv_zbe 93000
// CHECK-B-EXT: __riscv_zbf 93000
// CHECK-B-EXT: __riscv_zbm 93000
// CHECK-B-EXT: __riscv_zbp 93000
// CHECK-B-EXT: __riscv_zbr 93000
// CHECK-B-EXT: __riscv_zbs 93000
// CHECK-B-EXT: __riscv_zbt 93000

// RUN: %clang -target riscv32-unknown-linux-gnu -menable-experimental-extensions \
// RUN: -march=rv32izba0p93 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZBA-EXT %s
// RUN: %clang -target riscv64-unknown-linux-gnu -menable-experimental-extensions \
// RUN: -march=rv64izba0p93 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZBA-EXT %s
// CHECK-ZBA-NOT: __riscv_b
// CHECK-ZBA-EXT: __riscv_zba 93000

// RUN: %clang -target riscv32-unknown-linux-gnu -menable-experimental-extensions \
// RUN: -march=rv32izbb0p93 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZBB-EXT %s
// RUN: %clang -target riscv64-unknown-linux-gnu -menable-experimental-extensions \
// RUN: -march=rv64izbb0p93 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZBB-EXT %s
// CHECK-ZBB-NOT: __riscv_b
// CHECK-ZBB-EXT: __riscv_zbb 93000

// RUN: %clang -target riscv32-unknown-linux-gnu -menable-experimental-extensions \
// RUN: -march=rv32izbc0p93 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZBC-EXT %s
// RUN: %clang -target riscv64-unknown-linux-gnu -menable-experimental-extensions \
// RUN: -march=rv64izbc0p93 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZBC-EXT %s
// CHECK-ZBC-NOT: __riscv_b
// CHECK-ZBC-EXT: __riscv_zbc 93000

// RUN: %clang -target riscv32-unknown-linux-gnu -menable-experimental-extensions \
// RUN: -march=rv32izbe0p93 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZBE-EXT %s
// RUN: %clang -target riscv64-unknown-linux-gnu -menable-experimental-extensions \
// RUN: -march=rv64izbe0p93 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZBE-EXT %s
// CHECK-ZBE-NOT: __riscv_b
// CHECK-ZBE-EXT: __riscv_zbe 93000

// RUN: %clang -target riscv32-unknown-linux-gnu -menable-experimental-extensions \
// RUN: -march=rv32izbf0p93 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZBF-EXT %s
// RUN: %clang -target riscv64-unknown-linux-gnu -menable-experimental-extensions \
// RUN: -march=rv64izbf0p93 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZBF-EXT %s
// CHECK-ZBF-NOT: __riscv_b
// CHECK-ZBF-EXT: __riscv_zbf 93000

// RUN: %clang -target riscv32-unknown-linux-gnu -menable-experimental-extensions \
// RUN: -march=rv32izbm0p93 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZBM-EXT %s
// RUN: %clang -target riscv64-unknown-linux-gnu -menable-experimental-extensions \
// RUN: -march=rv64izbm0p93 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZBM-EXT %s
// CHECK-ZBM-NOT: __riscv_b
// CHECK-ZBM-EXT: __riscv_zbm 93000

// RUN: %clang -target riscv32-unknown-linux-gnu -menable-experimental-extensions \
// RUN: -march=rv32izbp0p93 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZBP-EXT %s
// RUN: %clang -target riscv64-unknown-linux-gnu -menable-experimental-extensions \
// RUN: -march=rv64izbp0p93 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZBP-EXT %s
// CHECK-ZBP-NOT: __riscv_b
// CHECK-ZBP-EXT: __riscv_zbp 93000

// RUN: %clang -target riscv32-unknown-linux-gnu -menable-experimental-extensions \
// RUN: -march=rv32izbr0p93 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZBR-EXT %s
// RUN: %clang -target riscv64-unknown-linux-gnu -menable-experimental-extensions \
// RUN: -march=rv64izbr0p93 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZBR-EXT %s
// CHECK-ZBR-NOT: __riscv_b
// CHECK-ZBR-EXT: __riscv_zbr 93000

// RUN: %clang -target riscv32-unknown-linux-gnu -menable-experimental-extensions \
// RUN: -march=rv32izbs0p93 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZBS-EXT %s
// RUN: %clang -target riscv64-unknown-linux-gnu -menable-experimental-extensions \
// RUN: -march=rv64izbs0p93 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZBS-EXT %s
// CHECK-ZBS-NOT: __riscv_b
// CHECK-ZBS-EXT: __riscv_zbs 93000

// RUN: %clang -target riscv32-unknown-linux-gnu -menable-experimental-extensions \
// RUN: -march=rv32izbt0p93 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZBT-EXT %s
// RUN: %clang -target riscv64-unknown-linux-gnu -menable-experimental-extensions \
// RUN: -march=rv64izbt0p93 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZBT-EXT %s
// CHECK-ZBT-NOT: __riscv_b
// CHECK-ZBT-EXT: __riscv_zbt 93000

// RUN: %clang -target riscv32-unknown-linux-gnu -menable-experimental-extensions \
// RUN: -march=rv32iv0p10 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-V-EXT %s
// RUN: %clang -target riscv64-unknown-linux-gnu -menable-experimental-extensions \
// RUN: -march=rv64iv0p10 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-V-EXT %s
// RUN: %clang -target riscv32-unknown-linux-gnu -menable-experimental-extensions \
// RUN: -march=rv32izvlsseg0p10 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-V-EXT %s
// RUN: %clang -target riscv64-unknown-linux-gnu -menable-experimental-extensions \
// RUN: -march=rv32izvlsseg0p10 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-V-EXT %s
// CHECK-V-EXT: __riscv_v 10000
// CHECK-V-EXT: __riscv_vector 1
// CHECK-V-EXT: __riscv_zvlsseg 10000

// RUN: %clang -target riscv32-unknown-linux-gnu -menable-experimental-extensions \
// RUN: -march=rv32izvamo0p10 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZVAMO-EXT %s
// RUN: %clang -target riscv64-unknown-linux-gnu -menable-experimental-extensions \
// RUN: -march=rv32izvamo0p10 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZVAMO-EXT %s
// CHECK-ZVAMO-EXT: __riscv_v 10000
// CHECK-ZVAMO-EXT: __riscv_vector 1
// CHECK-ZVAMO-EXT: __riscv_zvamo 10000
// CHECK-ZVAMO-EXT: __riscv_zvlsseg 10000

// RUN: %clang -target riscv32-unknown-linux-gnu -menable-experimental-extensions \
// RUN: -march=rv32izfh0p1 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZFH-EXT %s
// RUN: %clang -target riscv64-unknown-linux-gnu -menable-experimental-extensions \
// RUN: -march=rv64izfh0p1 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZFH-EXT %s
// CHECK-ZFH-EXT: __riscv_zfh 1000
