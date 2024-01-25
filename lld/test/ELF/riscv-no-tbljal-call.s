# REQUIRES: riscv

# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+relax -mattr=zcmt %s -o %t.rv32.o
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+relax -mattr=zcmt %s -o %t.rv64.o

# tbljal conversion
# RUN: ld.lld %t.rv32.o --relax-tbljal --defsym foo=0x150000 -o %t.rv32
# RUN: ld.lld %t.rv64.o --relax-tbljal --defsym foo=0x150000 -o %t.rv64

# jump table
# RUN: llvm-objdump -h %t.rv32 | FileCheck --check-prefix=JUMPTABLE32 %s
# RUN: llvm-objdump -h %t.rv64 | FileCheck --check-prefix=JUMPTABLE64 %s

# JUMPTABLE32:  2 .riscv.jvt    00000000 000110d4 TEXT
# JUMPTABLE64:  2 .riscv.jvt    00000000 0000000000011140 TEXT

.global _start
.p2align 3
_start:
  call foo
  tail foo_1
  tail foo_2
  tail foo_3

foo_1:
  nop

foo_2:
  nop

foo_3:
  nop

