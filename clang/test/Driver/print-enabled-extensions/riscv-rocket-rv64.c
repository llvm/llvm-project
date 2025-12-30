// REQUIRES: riscv-registered-target
// RUN: %clang --target=riscv64 --print-enabled-extensions -mcpu=rocket-rv64 | FileCheck --strict-whitespace %s

// Simple litmus test to check the frontend handling of this option is
// enabled.

// CHECK: Extensions enabled for the given RISC-V target
// CHECK-EMPTY:
// CHECK-NEXT: Name                 Version   Description
// CHECK-NEXT:     i                    2.1       'I' (Base Integer Instruction Set)
// CHECK-NEXT:     zicsr                2.0       'Zicsr' (CSRs)
// CHECK-NEXT:     zifencei             2.0       'Zifencei' (fence.i)
// CHECK-EMPTY:
