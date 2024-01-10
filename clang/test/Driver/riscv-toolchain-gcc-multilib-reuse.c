// RUN: %clang %s \
// RUN:   -target riscv64-unknown-elf \
// RUN:   --gcc-toolchain=%S/Inputs/multilib_riscv_elf_sdk \
// RUN:   --print-multi-directory \
// RUN:   -march=rv32imc -mabi=ilp32 \
// RUN:   | FileCheck -check-prefix=GCC-MULTI-LIB-REUSE-RV32IMC-ILP32 %s
// GCC-MULTI-LIB-REUSE-RV32IMC-ILP32: rv32im/ilp32
// GCC-MULTI-LIB-REUSE-RV32IMC-ILP32-NOT:  {{^.+$}}

// Check rv32imac won't reuse rv32im or rv32ic
// RUN: %clang %s \
// RUN:   -target riscv64-unknown-elf \
// RUN:   --gcc-toolchain=%S/Inputs/multilib_riscv_elf_sdk \
// RUN:   --print-multi-directory \
// RUN:   -march=rv32imac -mabi=ilp32 \
// RUN:   | FileCheck -check-prefix=GCC-MULTI-LIB-REUSE-RV32IMAC-ILP32 %s
// GCC-MULTI-LIB-REUSE-RV32IMAC-ILP32: rv32imac/ilp32
// GCC-MULTI-LIB-REUSE-RV32IMAC-ILP32--NOT: {{^.+$}}

// RUN: %clang %s \
// RUN:   -target riscv64-unknown-elf \
// RUN:   --gcc-toolchain=%S/Inputs/multilib_riscv_elf_sdk \
// RUN:   --print-multi-directory \
// RUN:   -march=rv32iac -mabi=ilp32 \
// RUN:   | FileCheck -check-prefix=GCC-MULTI-LIB-REUSE-RV32IAC-ILP32 %s
// GCC-MULTI-LIB-REUSE-RV32IAC-ILP32: rv32iac/ilp32
// GCC-MULTI-LIB-REUSE-RV32IAC-ILP32-NOT: {{^.+$}}

// RUN: %clang %s \
// RUN:   -target riscv64-unknown-elf \
// RUN:   --gcc-toolchain=%S/Inputs/multilib_riscv_elf_sdk \
// RUN:   --print-multi-directory \
// RUN:   -march=rv32imafdc -mabi=ilp32f \
// RUN:   | FileCheck -check-prefix=GCC-MULTI-LIB-REUSE-RV32IMAFDC-ILP32F %s
// GCC-MULTI-LIB-REUSE-RV32IMAFDC-ILP32F: rv32imafc/ilp32f
// GCC-MULTI-LIB-REUSE-RV32IMAFDC-ILP32F-NOT: {{^.+$}}

// RUN: %clang %s \
// RUN:   -target riscv64-unknown-elf \
// RUN:   --gcc-toolchain=%S/Inputs/multilib_riscv_elf_sdk \
// RUN:   --print-multi-directory \
// RUN:   -march=rv32imafdc -mabi=ilp32d \
// RUN:   | FileCheck -check-prefix=GCC-MULTI-LIB-REUSE-RV32IMAFDC-ILP32D %s
// GCC-MULTI-LIB-REUSE-RV32IMAFDC-ILP32D: .
// GCC-MULTI-LIB-REUSE-RV32IMAFDC-ILP32D-NOT: {{^.+$}}

// RUN: %clang %s \
// RUN:   -target riscv64-unknown-elf \
// RUN:   --gcc-toolchain=%S/Inputs/multilib_riscv_elf_sdk \
// RUN:   --print-multi-directory \
// RUN:   -march=rv64imafc -mabi=lp64 \
// RUN:   | FileCheck -check-prefix=GCC-MULTI-LIB-REUSE-RV64IMAFC-LP64 %s
// GCC-MULTI-LIB-REUSE-RV64IMAFC-LP64: rv64imac/lp64
// GCC-MULTI-LIB-REUSE-RV64IMAFC-LP64-NOT: {{^.+$}}

// RUN: %clang %s \
// RUN:   -target riscv64-unknown-elf \
// RUN:   --gcc-toolchain=%S/Inputs/multilib_riscv_elf_sdk \
// RUN:   --print-multi-directory \
// RUN:   -march=rv32imafc_zfh -mabi=ilp32 \
// RUN:   | FileCheck -check-prefix=GCC-MULTI-LIB-REUSE-RV32IMAFC_ZFH-ILP32 %s
// GCC-MULTI-LIB-REUSE-RV32IMAFC_ZFH-ILP32: rv32imac/ilp32
// GCC-MULTI-LIB-REUSE-RV32IMAFC_ZFH-ILP32-NOT: {{^.+$}}

// RUN: %clang %s \
// RUN:   -target riscv64-unknown-elf \
// RUN:   --gcc-toolchain=%S/Inputs/multilib_riscv_elf_sdk \
// RUN:   --print-multi-directory \
// RUN:   -march=rv32i_zvkb -mabi=ilp32 \
// RUN:   | FileCheck -check-prefix=GCC-MULTI-LIB-REUSE-RV32I_ZVKB-ILP32 %s
// GCC-MULTI-LIB-REUSE-RV32I_ZVKB-ILP32: rv32i/ilp32
// GCC-MULTI-LIB-REUSE-RV32I_ZVKB-ILP32-NOT: {{^.+$}}

// RUN: %clang %s \
// RUN:   -target riscv64-unknown-elf \
// RUN:   --gcc-toolchain=%S/Inputs/multilib_riscv_elf_sdk \
// RUN:   --print-multi-directory \
// RUN:   -march=rv64imfc -mabi=lp64 \
// RUN:   | FileCheck -check-prefix=GCC-MULTI-LIB-REUSE-RV64IMFC-LP64 %s
// GCC-MULTI-LIB-REUSE-RV64IMFC-LP64: .
// GCC-MULTI-LIB-REUSE-RV64IMFC-LP64-NOT: {{^.+$}}
