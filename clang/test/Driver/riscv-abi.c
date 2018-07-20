// RUN: %clang -target riscv32-unknown-elf %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-ILP32 %s
// RUN: %clang -target riscv32-unknown-elf %s -### -o %t.o -mabi=ilp32 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-ILP32 %s

// CHECK-ILP32: "-target-abi" "ilp32"

// TODO: ilp32f support.
// RUN: not %clang -target riscv32-unknown-elf %s -o %t.o -mabi=ilp32f 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-ILP32F %s

// CHECK-ILP32F: error: unknown target ABI 'ilp32f'

// TODO: ilp32d support.
// RUN: not %clang -target riscv32-unknown-elf %s -o %t.o -mabi=ilp32d 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-ILP32D %s

// CHECK-ILP32D: error: unknown target ABI 'ilp32d'

// RUN: not %clang -target riscv32-unknown-elf %s -o %t.o -mabi=lp64 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-RV32-LP64 %s

// CHECK-RV32-LP64: error: unknown target ABI 'lp64'

// RUN: %clang -target riscv64-unknown-elf %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-LP64 %s
// RUN: %clang -target riscv64-unknown-elf %s -### -o %t.o -mabi=lp64 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-LP64 %s

// CHECK-LP64: "-target-abi" "lp64"

// TODO: lp64f support.
// RUN: not %clang -target riscv64-unknown-elf %s -o %t.o -mabi=lp64f 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-LP64F %s

// CHECK-LP64F: error: unknown target ABI 'lp64f'

// TODO: lp64d support.
// RUN: not %clang -target riscv64-unknown-elf %s -o %t.o -mabi=lp64d 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-LP64D %s

// CHECK-LP64D: error: unknown target ABI 'lp64d'

// RUN: not %clang -target riscv64-unknown-elf %s -o %t.o -mabi=ilp32 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-RV64-ILP32 %s

// CHECK-RV64-ILP32: error: unknown target ABI 'ilp32'
