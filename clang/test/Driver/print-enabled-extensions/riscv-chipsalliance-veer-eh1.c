// REQUIRES: riscv-registered-target
// RUN: %clang --target=riscv32 -mcpu=chipsalliance-veer-eh1 --print-enabled-extensions | FileCheck %s

// CHECK: Extensions enabled for the given RISC-V target
// CHECK-EMPTY:
// CHECK-NEXT:     Name                 Version   Description
// CHECK-NEXT:     i                    2.1       'I' (Base Integer Instruction Set)
// CHECK-NEXT:     m                    2.0       'M' (Integer Multiplication and Division)
// CHECK-NEXT:     c                    2.0       'C' (Compressed Instructions)
// CHECK-NEXT:     zicsr                2.0       'Zicsr' (CSRs)
// CHECK-NEXT:     zifencei             2.0       'Zifencei' (fence.i)
// CHECK-NEXT:     zmmul                1.0       'Zmmul' (Integer Multiplication)
// CHECK-NEXT:     zca                  1.0       'Zca' (part of the C extension, excluding compressed floating point loads/stores)
// CHECK-EMPTY:
// CHECK-NEXT: Experimental extensions
// CHECK-EMPTY:
// CHECK-NEXT: ISA String: rv32i2p1_m2p0_c2p0_zicsr2p0_zifencei2p0_zmmul1p0_zca1p0
