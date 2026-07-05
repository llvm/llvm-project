// RUN: %clang %s -mllvm -pre-RA-sched=fast -c -o %t-fast.o 2>&1 | FileCheck --allow-empty %s
// RUN: %clang %s -mllvm -pre-RA-sched=linearize -c -o %t-linearize.o 2>&1 | FileCheck --allow-empty %s

// CHECK-NOT: clang (LLVM option parsing): for the --pre-RA-sched option: Cannot find option named
