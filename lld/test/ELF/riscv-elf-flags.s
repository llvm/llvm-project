# REQUIRES: riscv

# RUN: echo -n "BLOB" > %t.binary
# RUN: ld.lld -m elf64lriscv -b binary %t.binary -o %t.out
# RUN: llvm-readobj -h %t.out | FileCheck %s

# CHECK:      Flags [
# CHECK-NEXT: ]
