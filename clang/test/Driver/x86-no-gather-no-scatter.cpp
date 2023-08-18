/// Tests -mno-gather and -mno-scatter
// RUN: %clang -c -mno-gather -### %s 2>&1 | FileCheck --check-prefix=NOGATHER %s
// RUN: %clang_cl -c /Qgather- -### %s 2>&1 | FileCheck --check-prefix=NOGATHER %s
// NOGATHER: "-target-feature" "+prefer-no-gather"

// RUN: %clang -c -mno-scatter -### %s 2>&1 | FileCheck --check-prefix=NOSCATTER %s
// RUN: %clang_cl -c /Qscatter- -### %s 2>&1 | FileCheck --check-prefix=NOSCATTER %s
// NOSCATTER: "-target-feature" "+prefer-no-scatter"
