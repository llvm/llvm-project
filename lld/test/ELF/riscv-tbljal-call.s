# REQUIRES: riscv

# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+relax -mattr=zcmt %s -o %t.rv32.o
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+relax -mattr=zcmt %s -o %t.rv64.o

# tbljal conversion
# RUN: ld.lld %t.rv32.o --relax-tbljal --defsym foo=0x150000 --defsym foo_1=0x150010 --defsym foo_3=0x150030 -o %t.rv32
# RUN: ld.lld %t.rv64.o --relax-tbljal --defsym foo=0x150000 --defsym foo_1=0x150010 --defsym foo_3=0x150030 -o %t.rv64
# RUN: llvm-objdump -d -M no-aliases --mattr=zcmt --no-show-raw-insn %t.rv32 | FileCheck --check-prefix=TBLJAL32 %s
# RUN: llvm-objdump -d -M no-aliases --mattr=zcmt --no-show-raw-insn %t.rv64 | FileCheck --check-prefix=TBLJAL64 %s

# jump table
# RUN: llvm-objdump -j .riscv.jvt -s %t.rv32 | FileCheck --check-prefix=JUMPTABLE32 %s
# RUN: llvm-objdump -j .riscv.jvt -s %t.rv64 | FileCheck --check-prefix=JUMPTABLE64 %s

# TBLJAL32:      cm.jalt 0x20
# TBLJAL32-NEXT: cm.jalt 0x20
# TBLJAL32-NEXT: cm.jalt 0x20
# TBLJAL32-NEXT: cm.jalt 0x20
# TBLJAL32-NEXT: cm.jalt 0x20
# TBLJAL32-NEXT: cm.jalt 0x20
# TBLJAL32-NEXT: cm.jalt 0x20
# TBLJAL32-NEXT: cm.jalt 0x20
# TBLJAL32-NEXT: cm.jalt 0x20
# TBLJAL32-NEXT: cm.jalt 0x20
# TBLJAL32-NEXT: cm.jalt 0x20
# TBLJAL32-NEXT: cm.jalt 0x20
# TBLJAL32-NEXT: cm.jalt 0x20
# TBLJAL32-NEXT: cm.jalt 0x20
# TBLJAL32-NEXT: cm.jalt 0x20
# TBLJAL32-NEXT: cm.jalt 0x20
# TBLJAL32-NEXT: cm.jalt 0x20
# TBLJAL32-NEXT: cm.jalt 0x20
# TBLJAL32-NEXT: cm.jalt 0x20
# TBLJAL32-NEXT: cm.jalt 0x20
# TBLJAL32-NEXT: cm.jalt 0x20
# TBLJAL32-NEXT: cm.jt   0x2
# TBLJAL32-NEXT: cm.jt   0x1
# TBLJAL32-NEXT: cm.jt   0x1
# TBLJAL32-NEXT: cm.jt   0x1
# TBLJAL32-NEXT: cm.jt   0x0
# TBLJAL32-NEXT: c.j     0x110f6 <foo_2>
# TBLJAL32-NEXT: cm.jt   0x0
# TBLJAL32-NEXT: cm.jt   0x0
# TBLJAL32-NEXT: cm.jt   0x0
# TBLJAL32-NEXT: cm.jt   0x0

# TBLJAL64:      cm.jt   0x1
# TBLJAL64-NEXT: cm.jt   0x1
# TBLJAL64-NEXT: cm.jt   0x1
# TBLJAL64-NEXT: cm.jt   0x0
# TBLJAL64-NEXT: c.j     0x111e2 <foo_2>
# TBLJAL64-NEXT: cm.jt   0x0
# TBLJAL64-NEXT: cm.jt   0x0
# TBLJAL64-NEXT: cm.jt   0x0
# TBLJAL64-NEXT: cm.jt   0x0


# JUMPTABLE32:      30001500 10001500 00001500 00000000
# JUMPTABLE32-NEXT: 00000000 00000000 00000000 00000000
# JUMPTABLE32-NEXT: 00000000 00000000 00000000 00000000
# JUMPTABLE32-NEXT: 00000000 00000000 00000000 00000000
# JUMPTABLE32-NEXT: 00000000 00000000 00000000 00000000
# JUMPTABLE32-NEXT: 00000000 00000000 00000000 00000000
# JUMPTABLE32-NEXT: 00000000 00000000 00000000 00000000
# JUMPTABLE32-NEXT: 00000000 00000000 00000000 00000000
# JUMPTABLE32-NEXT: 00001500

# JUMPTABLE64:      30001500 00000000 10001500 00000000

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
  tail foo
  tail foo_1
  tail foo_1
  tail foo_1
  tail foo_3
  tail foo_2
  tail foo_3
  tail foo_3
  tail foo_3
  tail foo_3

foo_2:
  nop


