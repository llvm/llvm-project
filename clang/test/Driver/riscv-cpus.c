// Check target CPUs are correctly passed.

// RUN: %clang -target riscv32 -### -c %s 2>&1 -mcpu=rocket-rv32 | FileCheck -check-prefix=MCPU-ROCKET32 %s
// MCPU-ROCKET32: "-nostdsysteminc" "-target-cpu" "rocket-rv32"

// RUN: %clang -target riscv64 -### -c %s 2>&1 -mcpu=rocket-rv64 | FileCheck -check-prefix=MCPU-ROCKET64 %s
// MCPU-ROCKET64: "-nostdsysteminc" "-target-cpu" "rocket-rv64"
// MCPU-ROCKET64: "-target-feature" "+64bit"

// RUN: %clang -target riscv32 -### -c %s 2>&1 -mcpu=sifive-7-rv32 | FileCheck -check-prefix=MCPU-SIFIVE7-32 %s
// MCPU-SIFIVE7-32: "-nostdsysteminc" "-target-cpu" "sifive-7-rv32"

// RUN: %clang -target riscv64 -### -c %s 2>&1 -mcpu=sifive-7-rv64 | FileCheck -check-prefix=MCPU-SIFIVE7-64 %s
// MCPU-SIFIVE7-64: "-nostdsysteminc" "-target-cpu" "sifive-7-rv64"
// MCPU-SIFIVE7-64: "-target-feature" "+64bit"

// mcpu with default march
// RUN: %clang -target riscv64 -### -c %s 2>&1 -mcpu=sifive-u54 | FileCheck -check-prefix=MCPU-SIFIVE-U54 %s
// MCPU-SIFIVE-U54: "-nostdsysteminc" "-target-cpu" "sifive-u54"
// MCPU-SIFIVE-U54: "-target-feature" "+m" "-target-feature" "+a" "-target-feature" "+f" "-target-feature" "+d"
// MCPU-SIFIVE-U54: "-target-feature" "+c" "-target-feature" "+64bit"
// MCPU-SIFIVE-U54: "-target-abi" "lp64d"

// mcpu with mabi option
// RUN: %clang -target riscv64 -### -c %s 2>&1 -mcpu=sifive-u54 -mabi=lp64 | FileCheck -check-prefix=MCPU-ABI-SIFIVE-U54 %s
// MCPU-ABI-SIFIVE-U54: "-nostdsysteminc" "-target-cpu" "sifive-u54"
// MCPU-ABI-SIFIVE-U54: "-target-feature" "+m" "-target-feature" "+a" "-target-feature" "+f" "-target-feature" "+d"
// MCPU-ABI-SIFIVE-U54: "-target-feature" "+c" "-target-feature" "+64bit"
// MCPU-ABI-SIFIVE-U54: "-target-abi" "lp64"

// mcpu with default march
// RUN: %clang -target riscv64 -### -c %s 2>&1 -mcpu=sifive-e76 | FileCheck -check-prefix=MCPU-SIFIVE-E76 %s
// MCPU-SIFIVE-E76: "-nostdsysteminc" "-target-cpu" "sifive-e76"
// MCPU-SIFIVE-E76: "-target-feature" "+m" "-target-feature" "+a" "-target-feature" "+f" "-target-feature" "+d"
// MCPU-SIFIVE-E76: "-target-feature" "+c"
// MCPU-SIFIVE-E76: "-target-abi" "lp64d"

// mcpu with mabi option
// RUN: %clang -target riscv64 -### -c %s 2>&1 -mcpu=sifive-u74 -mabi=lp64 | FileCheck -check-prefix=MCPU-ABI-SIFIVE-U74 %s
// MCPU-ABI-SIFIVE-U74: "-nostdsysteminc" "-target-cpu" "sifive-u74"
// MCPU-ABI-SIFIVE-U74: "-target-feature" "+m" "-target-feature" "+a" "-target-feature" "+f" "-target-feature" "+d"
// MCPU-ABI-SIFIVE-U74: "-target-feature" "+c" "-target-feature" "+64bit"
// MCPU-ABI-SIFIVE-U74: "-target-abi" "lp64"

// march overwrite mcpu's default march
// RUN: %clang -target riscv32 -### -c %s 2>&1 -mcpu=sifive-e31 -march=rv32imc | FileCheck -check-prefix=MCPU-MARCH %s
// MCPU-MARCH: "-nostdsysteminc" "-target-cpu" "sifive-e31" "-target-feature" "+m" "-target-feature" "+c"
// MCPU-MARCH: "-target-abi" "ilp32"

// Check failed cases

// RUN: %clang -target riscv32 -### -c %s 2>&1 -mcpu=generic-rv321 | FileCheck -check-prefix=FAIL-MCPU-NAME %s
// FAIL-MCPU-NAME: error: the clang compiler does not support '-mcpu=generic-rv321'

// RUN: %clang -target riscv32 -### -c %s 2>&1 -mcpu=generic-rv32 -march=rv64i | FileCheck -check-prefix=MISMATCH-ARCH %s
// MISMATCH-ARCH: error: the clang compiler does not support '-mcpu=generic-rv32'

// RUN: %clang -target riscv32 -### -c %s 2>&1 -mcpu=generic-rv64 | FileCheck -check-prefix=MISMATCH-MCPU %s
// MISMATCH-MCPU: error: the clang compiler does not support '-mcpu=generic-rv64'
