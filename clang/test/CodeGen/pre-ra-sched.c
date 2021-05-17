// RUN: %clang %s -mllvm -pre-RA-sched=fast -c -o - | FileCheck %s
// RUN: %clang %s -mllvm -pre-RA-sched=linearize -c -o - | FileCheck %s

// CHECK-NOT: clang (LLVM option parsing): for the --pre-RA-sched option: Cannot find option named
