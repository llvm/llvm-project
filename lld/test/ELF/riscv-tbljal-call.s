# REQUIRES: riscv

# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+relax -mattr=+experimental-zcmt %s -o %t.rv32.o
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+relax -mattr=+experimental-zcmt %s -o %t.rv64.o

# tbljal conversion
# RUN: ld.lld %t.rv32.o -riscv-tbljal --defsym foo=_start+30 -o %t.rv32
# RUN: ld.lld %t.rv64.o -riscv-tbljal --defsym foo=_start+30 -o %t.rv64
# RUN: llvm-objdump -d -M no-aliases --mattr=+experimental-zcmt --no-show-raw-insn %t.rv32 | FileCheck --check-prefix=TBLJAL %s
# RUN: llvm-objdump -d -M no-aliases --mattr=+experimental-zcmt --no-show-raw-insn %t.rv64 | FileCheck --check-prefix=TBLJAL %s
# TBLJAL:      cm.jalt 66
# TBLJAL-NEXT: cm.jt   2
# TBLJAL-NEXT: cm.jalt 67
# TBLJAL-NEXT: cm.jalt 65
# TBLJAL-NEXT: cm.jalt 65
# TBLJAL-NEXT: cm.jalt 64
# TBLJAL-NEXT: cm.jalt 64
# TBLJAL-NEXT: cm.jalt 64
# TBLJAL-NEXT: cm.jt   3
# TBLJAL-NEXT: cm.jt   1
# TBLJAL-NEXT: cm.jt   1
# TBLJAL-NEXT: cm.jt   0
# TBLJAL-NEXT: cm.jt   0
# TBLJAL-NEXT: cm.jt   0

# Check the bounds of what would be out of range (for the first call) for other jump types.
# RUN: ld.lld %t.rv32.o -riscv-tbljal --defsym foo=_start+0x100000 -o %t-boundary.rv32
# RUN: ld.lld %t.rv64.o -riscv-tbljal --defsym foo=_start+0x100000 -o %t-boundary.rv64
# RUN: ld.lld %t.rv32.o --defsym foo=_start+0x100000 -o %t-oldboundary.rv32
# RUN: ld.lld %t.rv64.o --defsym foo=_start+0x100000 -o %t-oldboundary.rv64
# RUN: llvm-objdump -d -M no-aliases --mattr=+experimental-zcmt --no-show-raw-insn %t-boundary.rv32 | FileCheck --check-prefix=BOUNDARY %s
# RUN: llvm-objdump -d -M no-aliases --mattr=+experimental-zcmt --no-show-raw-insn %t-boundary.rv64 | FileCheck --check-prefix=BOUNDARY %s
# RUN: llvm-objdump -d -M no-aliases --mattr=+experimental-zcmt --no-show-raw-insn %t-oldboundary.rv32 | FileCheck --check-prefix=OLDBOUNDARY %s
# RUN: llvm-objdump -d -M no-aliases --mattr=+experimental-zcmt --no-show-raw-insn %t-oldboundary.rv64 | FileCheck --check-prefix=OLDBOUNDARY %s
# OLDBOUNDARY:      auipc  ra, 256
# OLDBOUNDARY-NEXT: jalr   ra, 0(ra)
# OLDBOUNDARY-NEXT: jal    zero, {{.*}} <foo>
# OLDBOUNDARY-NEXT: jal    ra, {{.*}} <foo_1>
# OLDBOUNDARY-NEXT: jal    ra, {{.*}} <foo_2>
# OLDBOUNDARY-NEXT: jal    ra, {{.*}} <foo_2>
# OLDBOUNDARY-NEXT: jal    ra, {{.*}} <foo_3>
# OLDBOUNDARY-NEXT: jal    ra, {{.*}} <foo_3>
# OLDBOUNDARY-NEXT: jal    ra, {{.*}} <foo_3>
# OLDBOUNDARY-NEXT: jal    zero, {{.*}} <foo_1>
# OLDBOUNDARY-NEXT: jal    zero, {{.*}} <foo_2>
# OLDBOUNDARY-NEXT: jal    zero, {{.*}} <foo_2>
# OLDBOUNDARY-NEXT: jal    zero, {{.*}} <foo_3>
# OLDBOUNDARY-NEXT: jal    zero, {{.*}} <foo_3>
# OLDBOUNDARY-NEXT: jal    zero, {{.*}} <foo_3>
# BOUNDARY:      cm.jalt 66
# BOUNDARY-NEXT: cm.jt   2
# BOUNDARY-NEXT: cm.jalt 67
# BOUNDARY-NEXT: cm.jalt 65
# BOUNDARY-NEXT: cm.jalt 65
# BOUNDARY-NEXT: cm.jalt 64
# BOUNDARY-NEXT: cm.jalt 64
# BOUNDARY-NEXT: cm.jalt 64
# BOUNDARY-NEXT: cm.jt   3
# BOUNDARY-NEXT: cm.jt   1
# BOUNDARY-NEXT: cm.jt   1
# BOUNDARY-NEXT: cm.jt   0
# BOUNDARY-NEXT: cm.jt   0
# BOUNDARY-NEXT: cm.jt   0

# Check relaxation works across output sections
#  echo 'SECTIONS { .text 0x100000 : { *(.text) } .foo : ALIGN(8) { foo = .; } }' > %t-cross-section.lds
#  ld.lld %t.rv32c.o %t-cross-section.lds -o %t-cross-section.rv32
#  ld.lld %t.rv64c.o %t-cross-section.lds -o %t-cross-section.rv64

.global _start
.p2align 3
_start:
  call foo
  tail foo

  call foo_1
  call foo_2
  call foo_2
  call foo_3
  call foo_3
  call foo_3
  tail foo_1
  tail foo_2
  tail foo_2
  tail foo_3
  tail foo_3
  tail foo_3

foo_1:
  nop

foo_2:
  nop

foo_3:
  nop

