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

// RUN: %clang -target riscv32 -### -c %s 2>&1 -mtune=rocket-rv32 | FileCheck -check-prefix=MTUNE-ROCKET32 %s
// MTUNE-ROCKET32: "-tune-cpu" "rocket-rv32"

// RUN: %clang -target riscv64 -### -c %s 2>&1 -mtune=rocket-rv64 | FileCheck -check-prefix=MTUNE-ROCKET64 %s
// MTUNE-ROCKET64: "-tune-cpu" "rocket-rv64"

// RUN: %clang -target riscv32 -### -c %s 2>&1 -mtune=sifive-7-rv32 | FileCheck -check-prefix=MTUNE-SIFIVE7-32 %s
// MTUNE-SIFIVE7-32: "-tune-cpu" "sifive-7-rv32"

// RUN: %clang -target riscv64 -### -c %s 2>&1 -mtune=sifive-7-rv64 | FileCheck -check-prefix=MTUNE-SIFIVE7-64 %s
// MTUNE-SIFIVE7-64: "-tune-cpu" "sifive-7-rv64"

// Check mtune alias CPU has resolved to the right CPU according XLEN.
// RUN: %clang -target riscv32 -### -c %s 2>&1 -mtune=generic | FileCheck -check-prefix=MTUNE-GENERIC-32 %s
// MTUNE-GENERIC-32: "-tune-cpu" "generic-rv32"

// RUN: %clang -target riscv64 -### -c %s 2>&1 -mtune=generic | FileCheck -check-prefix=MTUNE-GENERIC-64 %s
// MTUNE-GENERIC-64: "-tune-cpu" "generic-rv64"

// RUN: %clang -target riscv32 -### -c %s 2>&1 -mtune=rocket | FileCheck -check-prefix=MTUNE-ROCKET-32 %s
// MTUNE-ROCKET-32: "-tune-cpu" "rocket-rv32"

// RUN: %clang -target riscv64 -### -c %s 2>&1 -mtune=rocket | FileCheck -check-prefix=MTUNE-ROCKET-64 %s
// MTUNE-ROCKET-64: "-tune-cpu" "rocket-rv64"

// RUN: %clang -target riscv32 -### -c %s 2>&1 -mtune=sifive-7-series | FileCheck -check-prefix=MTUNE-SIFIVE7-SERIES-32 %s
// MTUNE-SIFIVE7-SERIES-32: "-tune-cpu" "sifive-7-rv32"

// RUN: %clang -target riscv64 -### -c %s 2>&1 -mtune=sifive-7-series | FileCheck -check-prefix=MTUNE-SIFIVE7-SERIES-64 %s
// MTUNE-SIFIVE7-SERIES-64: "-tune-cpu" "sifive-7-rv64"

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
// MCPU-SIFIVE-E76: "-target-feature" "+m" "-target-feature" "+a" "-target-feature" "+f"
// MCPU-SIFIVE-E76: "-target-feature" "+c"
// MCPU-SIFIVE-E76: "-target-abi" "ilp32"

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

// Check interaction between mcpu and mtune, mtune won't affect arch related
// target feature, but mcpu will.
//
// In this case, sifive-e31 is rv32imac, sifive-e76 is rv32imafc, so M-extension
// should not enabled.
//
// RUN: %clang -target riscv32 -### -c %s 2>&1 -mcpu=sifive-e31 -mtune=sifive-e76 | FileCheck -check-prefix=MTUNE-E31-MCPU-E76 %s
// MTUNE-E31-MCPU-E76: "-target-cpu" "sifive-e31"
// MTUNE-E31-MCPU-E76-NOT: "-target-feature" "+f"
// MTUNE-E31-MCPU-E76-SAME: "-target-feature" "+m"
// MTUNE-E31-MCPU-E76-SAME: "-target-feature" "+a"
// MTUNE-E31-MCPU-E76-SAME: "-target-feature" "+c"
// MTUNE-E31-MCPU-E76-SAME: "-tune-cpu" "sifive-e76"

// Check failed cases

// RUN: %clang -target riscv32 -### -c %s 2>&1 -mcpu=generic-rv321 | FileCheck -check-prefix=FAIL-MCPU-NAME %s
// FAIL-MCPU-NAME: error: the clang compiler does not support '-mcpu=generic-rv321'

// RUN: %clang -target riscv32 -### -c %s 2>&1 -mcpu=generic-rv32 -march=rv64i | FileCheck -check-prefix=MISMATCH-ARCH %s
// MISMATCH-ARCH: error: the clang compiler does not support '-mcpu=generic-rv32'

// RUN: %clang -target riscv32 -### -c %s 2>&1 -mcpu=generic-rv64 | FileCheck -check-prefix=MISMATCH-MCPU %s
// MISMATCH-MCPU: error: the clang compiler does not support '-mcpu=generic-rv64'
