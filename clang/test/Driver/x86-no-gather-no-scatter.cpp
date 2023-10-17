/// Tests -mno-gather and -mno-scatter
// RUN: %clang -target x86_64-unknown-linux-gnu -c -mno-gather -### %s 2>&1 | FileCheck --check-prefix=NOGATHER %s
// RUN: %clang_cl --target=x86_64-windows -c /Qgather- -### -- %s 2>&1 | FileCheck --check-prefix=NOGATHER %s
// NOGATHER: "-target-feature" "+prefer-no-gather"

// RUN: %clang -target x86_64-unknown-linux-gnu -c -mno-scatter -### %s 2>&1 | FileCheck --check-prefix=NOSCATTER %s
// RUN: %clang_cl --target=x86_64-windows -c /Qscatter- -### -- %s 2>&1 | FileCheck --check-prefix=NOSCATTER %s
// NOSCATTER: "-target-feature" "+prefer-no-scatter"
