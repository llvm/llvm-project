// RUN: %clang_cc1 -triple riscv32 -emit-llvm -o - %s \
// RUN:   | FileCheck -check-prefix=RV32I %s
// RUN: %clang_cc1 -triple riscv32 -target-feature +v -emit-llvm -o - %s \
// RUN:   | FileCheck -check-prefix=RV32IV %s
// RUN: %clang_cc1 -triple riscv64 -emit-llvm -o - %s \
// RUN:   | FileCheck -check-prefix=RV64I %s
// RUN: %clang_cc1 -triple riscv64 -target-feature +v -emit-llvm -o - %s \
// RUN:   | FileCheck -check-prefix=RV64IV %s

// RV32I:!{{[0-9]+}} = !{i32 6, !"riscv-isa", ![[ID:[0-9]+]]}
// RV32I:![[ID]] = !{!"rv32i2p1"}

// RV32IV:!{{[0-9]+}} = !{i32 6, !"riscv-isa", ![[ID:[0-9]+]]}
// RV32IV:![[ID]] = !{!"rv32i2p1_f2p2_d2p2_v1p0_zicsr2p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0"}

// RV64I:!{{[0-9]+}} = !{i32 6, !"riscv-isa", ![[ID:[0-9]+]]}
// RV64I:![[ID]] = !{!"rv64i2p1"}

// RV64IV:!{{[0-9]+}} = !{i32 6, !"riscv-isa", ![[ID:[0-9]+]]}
// RV64IV:![[ID]] = !{!"rv64i2p1_f2p2_d2p2_v1p0_zicsr2p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0"}
