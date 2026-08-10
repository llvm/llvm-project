// RUN: %clang --target=riscv32-unknown-elf %s -### 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-ILP32 %s
// RUN: %clang --target=riscv32-unknown-elf %s -### -mabi=ilp32 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-ILP32 %s
// RUN: %clang --target=riscv32-unknown-elf %s -### -march=rv32imc 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-ILP32 %s
// RUN: %clang --target=riscv32-unknown-elf -x assembler %s -### 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-ILP32 %s
// RUN: %clang --target=riscv32-unknown-elf -x assembler %s -### \
// RUN:   -mabi=ilp32 2>&1 | FileCheck -check-prefix=CHECK-ILP32 %s

// CHECK-ILP32: "-target-abi" "ilp32"

// RUN: %clang --target=riscv32-unknown-elf %s -### -march=rv32e -mabi=ilp32e 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-ILP32E %s
// RUN: %clang --target=riscv32-unknown-elf %s -### -mabi=ilp32e 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-ILP32E %s
// RUN: %clang --target=riscv32-unknown-elf %s -### -march=rv32e 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-ILP32E %s

// CHECK-ILP32E: "-target-abi" "ilp32e"

// RUN: %clang --target=riscv32-unknown-elf %s -### -march=rv32if -mabi=ilp32f 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-ILP32F %s
// RUN: %clang --target=riscv32-unknown-elf %s -### -mabi=ilp32f 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-ILP32F %s
// RUN: %clang --target=riscv32-unknown-elf %s -### -march=rv32if 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-ILP32F %s

// CHECK-ILP32F: "-target-abi" "ilp32f"

// RUN: %clang --target=riscv32-unknown-elf %s -### -march=rv32ifd -mabi=ilp32d 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-ILP32D %s
// RUN: %clang --target=riscv32-unknown-elf %s -### -march=rv32ifd 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-ILP32D %s
// RUN: %clang --target=riscv32-unknown-elf %s -### -march=rv32g 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-ILP32D %s
// RUN: %clang --target=riscv32-unknown-linux-gnu %s -### 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-ILP32D %s
// RUN: %clang --target=riscv32-unknown-linux-gnu -x assembler %s -### 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-ILP32D %s

// CHECK-ILP32D: "-target-abi" "ilp32d"

// RUN: not %clang --target=riscv32-unknown-elf %s -o %t.o -mabi=lp64 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-RV32-LP64 %s

// CHECK-RV32-LP64: error: unknown target ABI 'lp64'

// RUN: %clang --target=riscv64-unknown-elf %s -### 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-LP64 %s
// RUN: %clang --target=riscv64-unknown-elf %s -### -mabi=lp64 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-LP64 %s
// RUN: %clang --target=riscv64-unknown-elf %s -### -march=rv64imc 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-LP64 %s
// RUN: %clang --target=riscv64-unknown-elf -x assembler %s -### 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-LP64  %s
// RUN: %clang --target=riscv64-unknown-elf -x assembler %s -### \
// RUN:   -mabi=lp64 2>&1 | FileCheck -check-prefix=CHECK-LP64 %s

// CHECK-LP64: "-target-abi" "lp64"

// RUN:  %clang --target=riscv64-unknown-elf %s -### -march=rv64if -mabi=lp64f 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-LP64F %s
// RUN:  %clang --target=riscv64-unknown-elf %s -### -mabi=lp64f 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-LP64F %s
// RUN:  %clang --target=riscv64-unknown-elf %s -### -march=rv64if 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-LP64F %s

// CHECK-LP64F: "-target-abi" "lp64f"

// RUN: %clang --target=riscv64-unknown-elf %s -### -march=rv64ifd -mabi=lp64d 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-LP64D %s
// RUN: %clang --target=riscv64-unknown-elf %s -### -march=rv64ifd 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-LP64D %s
// RUN: %clang --target=riscv64-unknown-elf %s -### -march=rv64g 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-LP64D %s
// RUN: %clang --target=riscv64-unknown-linux-gnu %s -### 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-LP64D %s
// RUN: %clang --target=riscv64-unknown-linux-gnu -x assembler %s -### 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-LP64D  %s

// CHECK-LP64D: "-target-abi" "lp64d"

// RUN: not %clang --target=riscv64-unknown-elf %s -o %t.o -mabi=ilp32 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-RV64-ILP32 %s

// CHECK-RV64-ILP32: error: unknown target ABI 'ilp32'
