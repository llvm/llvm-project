/// Tests -mapx-inline-asm-use-gpr32
// RUN: %clang -target x86_64-unknown-linux-gnu -c -mapx-inline-asm-use-gpr32 -### %s 2>&1 | FileCheck --check-prefix=GPR32 %s
// GPR32: "-target-feature" "+inline-asm-use-gpr32"
