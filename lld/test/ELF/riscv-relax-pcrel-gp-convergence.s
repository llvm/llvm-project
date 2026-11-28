# REQUIRES: riscv
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+relax %s -o %t.32.o
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+relax %s -o %t.64.o
# RUN: ld.lld --relax-gp --defsym '__global_pointer$=0x10808' -Ttext=0x10000 %t.32.o -o %t.32
# RUN: ld.lld --relax-gp --defsym '__global_pointer$=0x10808' -Ttext=0x10000 %t.64.o -o %t.64
# RUN: llvm-objdump -d --no-show-raw-insn -M no-aliases %t.32 | FileCheck %s
# RUN: llvm-objdump -d --no-show-raw-insn -M no-aliases %t.64 | FileCheck %s

## Removing AUIPC moves target from gp-2048 to gp-2052, invalidating the
## relaxation. Restoring it brings target back into range.
# CHECK:      <_start>:
# CHECK-NEXT: auipc a0, 0x0
# CHECK-NEXT: addi a0, a0, 0x8
# CHECK:      <target>:
# CHECK-NEXT: jalr zero, 0x0(ra)

.globl _start
_start:
  lla a0, target
target:
  ret
