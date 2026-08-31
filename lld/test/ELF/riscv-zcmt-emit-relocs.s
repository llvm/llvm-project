# REQUIRES: riscv

# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+c,+zcmt,+relax %s -o %t.o
# RUN: ld.lld -e _start --emit-relocs --riscv-relax-zcmt %t.o -o %t
# RUN: llvm-readelf -r %t | FileCheck %s --check-prefix=RELOCS \
# RUN:   --implicit-check-not=R_RISCV_CALL \
# RUN:   --implicit-check-not=R_RISCV_CALL_PLT \
# RUN:   --implicit-check-not=R_RISCV_JAL \
# RUN:   --implicit-check-not=R_RISCV_RELAX
# RUN: llvm-objdump -d --no-show-raw-insn -M no-aliases %t | FileCheck %s \
# RUN:   --check-prefix=DIS

# RELOCS: Relocation section '.rela.text'

# DIS-LABEL: <_start>:
# DIS-NEXT: cm.jt 0
# DIS-NEXT: cm.jt 0
# DIS-NEXT: cm.jt 0
# DIS-NEXT: cm.jt 0
# DIS-NEXT: cm.jt 0

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
