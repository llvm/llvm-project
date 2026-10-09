# REQUIRES: riscv

# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+c,+zcmt,+relax a.s -o a.o
# RUN: ld.lld -T lds --riscv-relax-zcmt a.o -o a
# RUN: llvm-readelf -s a | FileCheck %s --check-prefix=SYM
# RUN: llvm-objdump -d --no-show-raw-insn -M no-aliases a | FileCheck %s --check-prefix=DIS

# SYM: 0000000000000000 0 NOTYPE LOCAL DEFAULT 1 callee
# DIS-LABEL: <_start>:
# DIS-NEXT: cm.jt 0
# DIS-NEXT: cm.jt 0

#--- lds
ENTRY(_start)
SECTIONS {
  . = 0;
  .text : { *(.text.callee) *(.text.start) }
}

#--- a.s
.attribute arch, "rv64i2p1_zicsr2p0_zca1p0_zcmt1p0"
.option rvc
.section .text.callee,"ax",@progbits
callee:
  ret
.section .text.start,"ax",@progbits
.space 4096
.globl _start
_start:
  .rept 5
  tail callee
  .endr
