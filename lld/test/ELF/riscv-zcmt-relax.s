# REQUIRES: riscv

# RUN: rm -rf %t && split-file %s %t && cd %t

# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+c,+zcmt,+relax jt.s -o jt.o
# RUN: ld.lld -e _start --riscv-relax-zcmt jt.o -o jt
# RUN: llvm-readelf -S -s jt | FileCheck %s --check-prefix=JT-SEC
# RUN: llvm-objdump -d --no-show-raw-insn -M no-aliases jt | FileCheck %s --check-prefix=JT-DIS

# JT-SEC: .riscv.jvt
# JT-SEC-SAME: PROGBITS
# JT-SEC-SAME: AX
# JT-SEC-SAME: 64
# JT-SEC: __jvt_base$
# JT-DIS-LABEL: <_start>:
# JT-DIS-NEXT: cm.jt 0x0
# JT-DIS-NEXT: cm.jt 0x0
# JT-DIS-NEXT: cm.jt 0x0
# JT-DIS-NEXT: cm.jt 0x0
# JT-DIS-NEXT: cm.jt 0x0
# JT-DIS-NOT: cm.jt
# JT-DIS-LABEL: <callee>:

# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+c,+zcmt,+relax jalt.s -o jalt.o
# RUN: ld.lld -e _start --riscv-relax-zcmt jalt.o -o jalt
# RUN: llvm-readelf -S -s jalt | FileCheck %s --check-prefix=JALT-SEC
# RUN: llvm-objdump -d --no-show-raw-insn -M no-aliases jalt | FileCheck %s --check-prefix=JALT-DIS

# JALT-SEC: .riscv.jvt
# JALT-SEC-SAME: 000084
# JALT-SEC: __jvt_base$
# JALT-DIS-LABEL: <_start>:
# JALT-DIS-COUNT-67: cm.jalt 0x20
# JALT-DIS-NOT: cm.jalt
# JALT-DIS-LABEL: <callee>:

#--- jt.s
.attribute arch, "rv64i2p1_zicsr2p0_zca1p0_zcmt1p0"
.option rvc
.globl _start
_start:
  .rept 5
  tail callee
  .endr
  .space 4096
callee:
  ret

#--- jalt.s
.attribute arch, "rv32i2p1_zicsr2p0_zca1p0_zcmt1p0"
.option rvc
.globl _start
_start:
  .rept 67
  call callee
  .endr
  .space 4096
callee:
  ret
