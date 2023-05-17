// RUN: %clang -target riscv32-unknown-elf %s -S -emit-llvm -o - \
// RUN:   | FileCheck %s -check-prefix=RV32-DEFAULT
// RUN: %clang -target riscv32-unknown-elf %s -S -emit-llvm -G4 -o - \
// RUN:   | FileCheck %s -check-prefix=RV32-G4
// RUN: %clang -target riscv32-unknown-elf %s -S -emit-llvm -msmall-data-limit=0 -o - \
// RUN:   | FileCheck %s -check-prefix=RV32-S0
// RUN: %clang -target riscv32-unknown-elf %s -S -emit-llvm -msmall-data-limit=2 -G4 -o - \
// RUN:   | FileCheck %s -check-prefix=RV32-S2G4
// RUN: %clang -target riscv32-unknown-elf %s -S -emit-llvm -msmall-data-threshold=16 -o - \
// RUN:   | FileCheck %s -check-prefix=RV32-T16
// RUN: %clang -target riscv32-unknown-elf %s -S -emit-llvm -fpic -o - \
// RUN:   | FileCheck %s -check-prefix=RV32-PIC

// RUN: %clang -target riscv64-unknown-elf %s -S -emit-llvm -o - \
// RUN:   | FileCheck %s -check-prefix=RV64-DEFAULT
// RUN: %clang -target riscv64-unknown-elf %s -S -emit-llvm -G4 -o - \
// RUN:   | FileCheck %s -check-prefix=RV64-G4
// RUN: %clang -target riscv64-unknown-elf %s -S -emit-llvm -msmall-data-limit=0 -o - \
// RUN:   | FileCheck %s -check-prefix=RV64-S0
// RUN: %clang -target riscv64-unknown-elf %s -S -emit-llvm -msmall-data-limit=2 -G4 -o - \
// RUN:   | FileCheck %s -check-prefix=RV64-S2G4
// RUN: %clang -target riscv64-unknown-elf %s -S -emit-llvm -msmall-data-threshold=16 -o - \
// RUN:   | FileCheck %s -check-prefix=RV64-T16
// RUN: %clang -target riscv64-unknown-elf %s -S -emit-llvm -fpic -o - \
// RUN:   | FileCheck %s -check-prefix=RV64-PIC
// RUN: %clang -target riscv64-unknown-elf %s -S -emit-llvm -mcmodel=large -o - \
// RUN:   | FileCheck %s -check-prefix=RV64-LARGE

void test(void) {}

// RV32-DEFAULT: !{i32 8, !"SmallDataLimit", i32 8}
// RV32-G4:      !{i32 8, !"SmallDataLimit", i32 4}
// RV32-S0:      !{i32 8, !"SmallDataLimit", i32 0}
// RV32-S2G4:    !{i32 8, !"SmallDataLimit", i32 4}
// RV32-T16:     !{i32 8, !"SmallDataLimit", i32 16}
// RV32-PIC:     !{i32 8, !"SmallDataLimit", i32 0}

// RV64-DEFAULT: !{i32 8, !"SmallDataLimit", i32 8}
// RV64-G4:      !{i32 8, !"SmallDataLimit", i32 4}
// RV64-S0:      !{i32 8, !"SmallDataLimit", i32 0}
// RV64-S2G4:    !{i32 8, !"SmallDataLimit", i32 4}
// RV64-T16:     !{i32 8, !"SmallDataLimit", i32 16}
// RV64-PIC:     !{i32 8, !"SmallDataLimit", i32 0}
// RV64-LARGE:   !{i32 8, !"SmallDataLimit", i32 0}

// The value will be passed by module flag instead of target feature.
// RV32-S0-NOT: +small-data-limit=
// RV64-S0-NOT: +small-data-limit=
