// RUN: %clang -target riscv32-unknown-elf -march=rv32i -### %s -fsyntax-only 2>&1 | FileCheck %s
// RUN: %clang -target riscv32-unknown-elf -march=rv32im -### %s -fsyntax-only 2>&1 | FileCheck %s
// RUN: %clang -target riscv32-unknown-elf -march=rv32ima -### %s -fsyntax-only 2>&1 | FileCheck %s
// RUN: %clang -target riscv32-unknown-elf -march=rv32imaf -### %s -fsyntax-only 2>&1 | FileCheck %s
// RUN: %clang -target riscv32-unknown-elf -march=rv32imafd -### %s -fsyntax-only 2>&1 | FileCheck %s

// RUN: %clang -target riscv32-unknown-elf -march=rv32ic -### %s -fsyntax-only 2>&1 | FileCheck %s
// RUN: %clang -target riscv32-unknown-elf -march=rv32imc -### %s -fsyntax-only 2>&1 | FileCheck %s
// RUN: %clang -target riscv32-unknown-elf -march=rv32imac -### %s -fsyntax-only 2>&1 | FileCheck %s
// RUN: %clang -target riscv32-unknown-elf -march=rv32imafc -### %s -fsyntax-only 2>&1 | FileCheck %s
// RUN: %clang -target riscv32-unknown-elf -march=rv32imafdc -### %s -fsyntax-only 2>&1 | FileCheck %s

// RUN: %clang -target riscv32-unknown-elf -march=rv32ia -### %s -fsyntax-only 2>&1 | FileCheck %s
// RUN: %clang -target riscv32-unknown-elf -march=rv32iaf -### %s -fsyntax-only 2>&1 | FileCheck %s
// RUN: %clang -target riscv32-unknown-elf -march=rv32iafd -### %s -fsyntax-only 2>&1 | FileCheck %s

// RUN: %clang -target riscv32-unknown-elf -march=rv32iac -### %s -fsyntax-only 2>&1 | FileCheck %s
// RUN: %clang -target riscv32-unknown-elf -march=rv32iafc -### %s -fsyntax-only 2>&1 | FileCheck %s
// RUN: %clang -target riscv32-unknown-elf -march=rv32iafdc -### %s -fsyntax-only 2>&1 | FileCheck %s

// RUN: %clang -target riscv32-unknown-elf -march=rv32g -### %s -fsyntax-only 2>&1 | FileCheck %s
// RUN: %clang -target riscv32-unknown-elf -march=rv32gc -### %s -fsyntax-only 2>&1 | FileCheck %s

// RUN: %clang -target riscv64-unknown-elf -march=rv64i -### %s -fsyntax-only 2>&1 | FileCheck %s
// RUN: %clang -target riscv64-unknown-elf -march=rv64im -### %s -fsyntax-only 2>&1 | FileCheck %s
// RUN: %clang -target riscv64-unknown-elf -march=rv64ima -### %s -fsyntax-only 2>&1 | FileCheck %s
// RUN: %clang -target riscv64-unknown-elf -march=rv64imaf -### %s -fsyntax-only 2>&1 | FileCheck %s
// RUN: %clang -target riscv64-unknown-elf -march=rv64imafd -### %s -fsyntax-only 2>&1 | FileCheck %s

// RUN: %clang -target riscv64-unknown-elf -march=rv64ic -### %s -fsyntax-only 2>&1 | FileCheck %s
// RUN: %clang -target riscv64-unknown-elf -march=rv64imc -### %s -fsyntax-only 2>&1 | FileCheck %s
// RUN: %clang -target riscv64-unknown-elf -march=rv64imac -### %s -fsyntax-only 2>&1 | FileCheck %s
// RUN: %clang -target riscv64-unknown-elf -march=rv64imafc -### %s -fsyntax-only 2>&1 | FileCheck %s
// RUN: %clang -target riscv64-unknown-elf -march=rv64imafdc -### %s -fsyntax-only 2>&1 | FileCheck %s

// RUN: %clang -target riscv64-unknown-elf -march=rv64ia -### %s -fsyntax-only 2>&1 | FileCheck %s
// RUN: %clang -target riscv64-unknown-elf -march=rv64iaf -### %s -fsyntax-only 2>&1 | FileCheck %s
// RUN: %clang -target riscv64-unknown-elf -march=rv64iafd -### %s -fsyntax-only 2>&1 | FileCheck %s

// RUN: %clang -target riscv64-unknown-elf -march=rv64iac -### %s -fsyntax-only 2>&1 | FileCheck %s
// RUN: %clang -target riscv64-unknown-elf -march=rv64iafc -### %s -fsyntax-only 2>&1 | FileCheck %s
// RUN: %clang -target riscv64-unknown-elf -march=rv64iafdc -### %s -fsyntax-only 2>&1 | FileCheck %s

// RUN: %clang -target riscv64-unknown-elf -march=rv64g -### %s -fsyntax-only 2>&1 | FileCheck %s
// RUN: %clang -target riscv64-unknown-elf -march=rv64gc -### %s -fsyntax-only 2>&1 | FileCheck %s

// CHECK-NOT: error: invalid arch name '

// RUN: %clang -target riscv32-unknown-elf -march=rv32 -### %s -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32 %s
// RV32: error: invalid arch name 'rv32'

// RUN: %clang -target riscv32-unknown-elf -march=rv32m -### %s -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32M %s
// RV32M: error: invalid arch name 'rv32m'

// RUN: %clang -target riscv32-unknown-elf -march=rv32id -### %s -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32ID %s
// RV32ID: error: invalid arch name 'rv32id'

// RUN: %clang -target riscv32-unknown-elf -march=rv32l -### %s -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32L %s
// RV32L: error: invalid arch name 'rv32l'

// RUN: %clang -target riscv32-unknown-elf -march=rv32imadf -### %s -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32IMADF %s
// RV32IMADF: error: invalid arch name 'rv32imadf'

// RUN: %clang -target riscv32-unknown-elf -march=rv32imm -### %s -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32IMM %s
// RV32IMM: error: invalid arch name 'rv32imm'

// RUN: %clang -target riscv32-unknown-elf -march=RV32I -### %s -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32I-UPPER %s
// RV32I-UPPER: error: invalid arch name 'RV32I'

// RUN: %clang -target riscv64-unknown-elf -march=rv64 -### %s -fsyntax-only 2>&1 | FileCheck -check-prefix=RV64 %s
// RV64: error: invalid arch name 'rv64'

// RUN: %clang -target riscv64-unknown-elf -march=rv64m -### %s -fsyntax-only 2>&1 | FileCheck -check-prefix=RV64M %s
// RV64M: error: invalid arch name 'rv64m'

// RUN: %clang -target riscv64-unknown-elf -march=rv64id -### %s -fsyntax-only 2>&1 | FileCheck -check-prefix=RV64ID %s
// RV64ID: error: invalid arch name 'rv64id'

// RUN: %clang -target riscv64-unknown-elf -march=rv64l -### %s -fsyntax-only 2>&1 | FileCheck -check-prefix=RV64L %s
// RV64L: error: invalid arch name 'rv64l'

// RUN: %clang -target riscv64-unknown-elf -march=rv64imadf -### %s -fsyntax-only 2>&1 | FileCheck -check-prefix=RV64IMADF %s
// RV64IMADF: error: invalid arch name 'rv64imadf'

// RUN: %clang -target riscv64-unknown-elf -march=rv64imm -### %s -fsyntax-only 2>&1 | FileCheck -check-prefix=RV64IMM %s
// RV64IMM: error: invalid arch name 'rv64imm'

// RUN: %clang -target riscv64-unknown-elf -march=RV64I -### %s -fsyntax-only 2>&1 | FileCheck -check-prefix=RV64I-UPPER %s
// RV64I-UPPER: error: invalid arch name 'RV64I'
