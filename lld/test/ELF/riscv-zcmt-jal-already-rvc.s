# REQUIRES: riscv

# RUN: rm -rf %t && split-file %s %t && cd %t

# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+c,+zcmt,+relax tail64.s -o tail64.o
# RUN: ld.lld -e _start --riscv-relax-zcmt tail64.o -o tail64
# RUN: llvm-readelf -S -s tail64 | FileCheck /dev/null \
# RUN:   --implicit-check-not=.riscv.jvt --implicit-check-not=__jvt_base$
# RUN: llvm-objdump -d --no-show-raw-insn -M no-aliases tail64 \
# RUN:   | FileCheck /dev/null --implicit-check-not=cm.jt \
# RUN:     --implicit-check-not=cm.jalt

# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+c,+zcmt,+relax call32.s -o call32.o
# RUN: ld.lld -e _start --riscv-relax-zcmt call32.o -o call32
# RUN: llvm-readelf -S -s call32 | FileCheck /dev/null \
# RUN:   --implicit-check-not=.riscv.jvt --implicit-check-not=__jvt_base$
# RUN: llvm-objdump -d --no-show-raw-insn -M no-aliases call32 \
# RUN:   | FileCheck /dev/null --implicit-check-not=cm.jt \
# RUN:     --implicit-check-not=cm.jalt

#--- tail64.s
.attribute arch, "rv64i2p1_zicsr2p0_zca1p0_zcmt1p0"
.option rvc
.globl _start
_start:
  .rept 5
  tail callee
  .endr
callee:
  ret

#--- call32.s
.attribute arch, "rv32i2p1_zicsr2p0_zca1p0_zcmt1p0"
.option rvc
.globl _start
_start:
  .rept 67
  call callee
  .endr
callee:
  ret
