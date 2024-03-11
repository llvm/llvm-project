# REQUIRES: riscv

# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+relax -mattr=zcmt %s -o %t.rv32.o
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+relax -mattr=zcmt %s -o %t.rv64.o

# tbljal conversion
# RUN: ld.lld %t.rv32.o --relax-tbljal --defsym foo=0x150000 -o %t.rv32
# RUN: ld.lld %t.rv64.o --relax-tbljal --defsym foo=0x150000 -o %t.rv64
# RUN: llvm-objdump -d -M no-aliases --mattr=zcmt --no-show-raw-insn %t.rv32 | FileCheck --check-prefix=TBLJAL32 %s
# RUN: llvm-objdump -d -M no-aliases --mattr=zcmt --no-show-raw-insn %t.rv64 | FileCheck --check-prefix=TBLJAL64 %s

# jump table
# RUN: llvm-objdump -j .riscv.jvt -s %t.rv32 | FileCheck --check-prefix=JUMPTABLE32 %s
# RUN: llvm-objdump -j .riscv.jvt -s %t.rv64 | FileCheck --check-prefix=JUMPTABLE64 %s

# TBLJAL32:      cm.jalt 32
# TBLJAL32-NEXT: cm.jalt 32
# TBLJAL32-NEXT: cm.jalt 32
# TBLJAL32-NEXT: cm.jalt 32
# TBLJAL32-NEXT: cm.jalt 32
# TBLJAL32-NEXT: cm.jalt 32
# TBLJAL32-NEXT: cm.jalt 32
# TBLJAL32-NEXT: cm.jalt 32
# TBLJAL32-NEXT: cm.jalt 32
# TBLJAL32-NEXT: cm.jalt 32
# TBLJAL32-NEXT: cm.jalt 32
# TBLJAL32-NEXT: cm.jalt 32
# TBLJAL32-NEXT: cm.jalt 32
# TBLJAL32-NEXT: cm.jalt 32
# TBLJAL32-NEXT: cm.jalt 32
# TBLJAL32-NEXT: cm.jalt 32
# TBLJAL32-NEXT: cm.jalt 32
# TBLJAL32-NEXT: cm.jalt 32
# TBLJAL32-NEXT: cm.jalt 32
# TBLJAL32-NEXT: cm.jalt 32
# TBLJAL32-NEXT: cm.jalt 32
# TBLJAL32-NEXT: cm.jt   1
# TBLJAL32-NEXT: cm.jt   1
# TBLJAL32-NEXT: cm.jt   1
# TBLJAL32-NEXT: cm.jt   1
# TBLJAL32-NEXT: cm.jt   0
# TBLJAL32-NEXT: jal     zero, 0x110fa
# TBLJAL32-NEXT: cm.jt   0
# TBLJAL32-NEXT: cm.jt   0
# TBLJAL32-NEXT: cm.jt   0
# TBLJAL32-NEXT: cm.jt   0

# TBLJAL64:      cm.jt   1
# TBLJAL64-NEXT: cm.jt   1
# TBLJAL64-NEXT: cm.jt   1
# TBLJAL64-NEXT: cm.jt   1
# TBLJAL64-NEXT: cm.jt   0
# TBLJAL64-NEXT: jal     zero, 0x111e0
# TBLJAL64-NEXT: cm.jt   0
# TBLJAL64-NEXT: cm.jt   0
# TBLJAL64-NEXT: cm.jt   0
# TBLJAL64-NEXT: cm.jt   0


# JUMPTABLE32:      fc100100 f8100100 00000000 00000000
# JUMPTABLE32-NEXT: 00000000 00000000 00000000 00000000
# JUMPTABLE32-NEXT: 00000000 00000000 00000000 00000000
# JUMPTABLE32-NEXT: 00000000 00000000 00000000 00000000
# JUMPTABLE32-NEXT: 00000000 00000000 00000000 00000000
# JUMPTABLE32-NEXT: 00000000 00000000 00000000 00000000
# JUMPTABLE32-NEXT: 00000000 00000000 00000000 00000000
# JUMPTABLE32-NEXT: 00000000 00000000 00000000 00000000
# JUMPTABLE32-NEXT: 00001500

# JUMPTABLE64:      e2110100 00000000 de110100 00000000

.global _start
.p2align 3
_start:
  call foo
  call foo
  call foo
  call foo
  call foo
  call foo
  call foo
  call foo
  call foo
  call foo
  call foo
  call foo
  call foo
  call foo
  call foo
  call foo
  call foo
  call foo
  call foo
  call foo
  call foo
  tail foo_1
  tail foo_1
  tail foo_1
  tail foo_1
  tail foo_3
  tail foo_2
  tail foo_3
  tail foo_3
  tail foo_3
  tail foo_3

foo_1:
  nop

foo_2:
  nop

foo_3:
  nop

