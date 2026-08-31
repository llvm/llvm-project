# REQUIRES: riscv

# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+c,+zcmt,+relax a.s -o a.o
# RUN: ld.lld -T lds --riscv-relax-zcmt a.o -o a
# RUN: llvm-readelf -S -s a | FileCheck %s --check-prefix=SEC \
# RUN:   --implicit-check-not=.riscv.jvt --implicit-check-not=__jvt_base$
# RUN: llvm-objdump -d --no-show-raw-insn -M no-aliases a | FileCheck %s \
# RUN:   --check-prefix=DIS --implicit-check-not=cm.jt

# SEC: .text
# SEC-DAG: callee
# SEC-DAG: _start

# DIS-LABEL: <_start>:
# DIS-NEXT: jal zero, {{.*}} <callee>

#--- lds
ENTRY(_start)
SECTIONS {
  .text : { *(.riscv.jvt) *(.text) }
}

#--- a.s
.attribute arch, "rv64i2p1_zicsr2p0_zca1p0_zcmt1p0"
.option rvc
.globl _start
_start:
  tail callee
  .space 4096
callee:
  ret
