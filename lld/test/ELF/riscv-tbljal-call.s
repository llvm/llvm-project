# REQUIRES: riscv

# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+relax -mattr=+experimental-zcmt %s -o %t.rv32.o
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+relax -mattr=+experimental-zcmt %s -o %t.rv64.o

# tbljal conversion
# RUN: ld.lld %t.rv32.o --riscv-tbljal --defsym foo=_start+0x150000 -o %t.rv32
# RUN: ld.lld %t.rv64.o --riscv-tbljal --defsym foo=_start+0x150000 -o %t.rv64
# RUN: llvm-objdump -d -M no-aliases --mattr=+experimental-zcmt --no-show-raw-insn %t.rv32 | FileCheck --check-prefix=TBLJAL32 %s
# RUN: llvm-objdump -d -M no-aliases --mattr=+experimental-zcmt --no-show-raw-insn %t.rv64 | FileCheck --check-prefix=TBLJAL64 %s
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
# TBLJAL32-NEXT: cm.jt   2
# TBLJAL32-NEXT: cm.jt   2
# TBLJAL32-NEXT: cm.jt   1
# TBLJAL32-NEXT: cm.jt   1
# TBLJAL32-NEXT: cm.jt   0
# TBLJAL32-NEXT: cm.jt   0
# TBLJAL32-NEXT: cm.jt   0
# TBLJAL32-NEXT: cm.jt   0

# TBLJAL64:      cm.jt   0
# TBLJAL64-NEXT: cm.jt   0
# TBLJAL64-NEXT: cm.jt   0
# TBLJAL64-NEXT: cm.jt   0


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
  tail foo_1
  tail foo_1
  tail foo_2
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

