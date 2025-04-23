// Check target CPUs are correctly passed.

// RUN: %clang --target=mips64 -### -c %s 2>&1 -mcpu=i6400 -mmsa | FileCheck -check-prefix=MCPU-I6400 %s
// MCPU-I6400: "-target-cpu" "i6400"
// MCPU-I6400-SAME: "-target-feature" "+msa"

// RUN: %clang --target=mips64 -### -c %s 2>&1 -mcpu=i6500 -mmsa | FileCheck -check-prefix=MCPU-I6500 %s
// MCPU-I6500: "-target-cpu" "i6500"
// MCPU-I6500-SAME: "-target-feature" "+msa"
