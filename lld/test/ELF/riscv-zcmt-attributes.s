# REQUIRES: riscv

# RUN: rm -rf %t && split-file %s %t && cd %t

# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+c,+relax noattr.s -o noattr.o
# RUN: ld.lld -e _start --riscv-relax-zcmt noattr.o -o noattr
# RUN: llvm-objdump -d --no-show-raw-insn -M no-aliases noattr | FileCheck %s --check-prefix=JUMP
# RUN: llvm-readelf -S -s noattr | FileCheck %s --check-prefix=HAS-JVT
# RUN: llvm-readelf -A noattr | FileCheck %s --check-prefix=ATTR

# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+c,+relax noattr-nocandidate.s -o noattr-nocandidate.o
# RUN: ld.lld -e _start --riscv-relax-zcmt noattr-nocandidate.o -o noattr-nocandidate
# RUN: llvm-readelf -h -S -l -s -A noattr-nocandidate | FileCheck %s \
# RUN:   --check-prefix=NO-ATTR --implicit-check-not=.riscv.attributes \
# RUN:   --implicit-check-not=RISCV_ATTRIBUTES --implicit-check-not=__jvt_base$ \
# RUN:   --implicit-check-not="TagName: arch"

# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+c,+relax nozcmt.s -o nozcmt.o
# RUN: ld.lld -e _start --riscv-relax-zcmt nozcmt.o -o nozcmt
# RUN: llvm-objdump -d --no-show-raw-insn -M no-aliases nozcmt | FileCheck %s --check-prefix=JUMP
# RUN: llvm-readelf -S -s nozcmt | FileCheck %s --check-prefix=HAS-JVT
# RUN: llvm-readelf -A nozcmt | FileCheck %s --check-prefix=ATTR

# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+c,+zcmt,+relax candidate.s -o candidate.o
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+c,+relax support-nozcmt.s -o support-nozcmt.o
# RUN: ld.lld -e _start --riscv-relax-zcmt candidate.o support-nozcmt.o -o mixed
# RUN: llvm-objdump -d --no-show-raw-insn -M no-aliases mixed | FileCheck %s --check-prefix=JUMP
# RUN: llvm-readelf -S -s mixed | FileCheck %s --check-prefix=HAS-JVT
# RUN: llvm-readelf -A mixed | FileCheck %s --check-prefix=ATTR

# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+c,+relax compatible-nocandidate.s -o compatible-nocandidate.o
# RUN: ld.lld -e _start --riscv-relax-zcmt compatible-nocandidate.o -o compatible-nocandidate
# RUN: llvm-readelf -S -s compatible-nocandidate | FileCheck /dev/null \
# RUN:   --implicit-check-not=.riscv.jvt --implicit-check-not=__jvt_base$

# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+c,+zcmt,+relax zcd.s -o zcd.o
# RUN: ld.lld -e _start --riscv-relax-zcmt zcd.o -o zcd 2>&1 | FileCheck %s --check-prefix=ZCD
# RUN: llvm-readelf -S -s zcd | FileCheck /dev/null \
# RUN:   --implicit-check-not=.riscv.jvt --implicit-check-not=__jvt_base$

# JUMP: cm.jt
# HAS-JVT: .riscv.jvt
# HAS-JVT: __jvt_base$
# ATTR: TagName: arch
# ATTR-NEXT: Value:
# ATTR-SAME: zicsr
# ATTR-SAME: zca
# ATTR-SAME: zcmt
# NO-ATTR: ELF Header:
# ZCD: warning: --riscv-relax-zcmt is disabled because
# ZCD-SAME: advertises incompatible Zcd/compressed-FP profile

#--- noattr.s
.option rvc
.globl _start
_start:
  .rept 5
  tail callee
  .endr
  .space 4096
callee:
  ret

#--- noattr-nocandidate.s
.option rvc
.globl _start
_start:
  ret

#--- nozcmt.s
.attribute arch, "rv64i2p1_zicsr2p0_zca1p0"
.option rvc
.globl _start
_start:
  .rept 5
  tail callee
  .endr
  .space 4096
callee:
  ret

#--- candidate.s
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

#--- support-nozcmt.s
.attribute arch, "rv64i2p1_zca1p0"
.option rvc
.globl support
support:
  ret

#--- compatible-nocandidate.s
.attribute arch, "rv64i2p1_zca1p0"
.option rvc
.globl _start
_start:
  ret

#--- zcd.s
.attribute arch, "rv64i2p1_zicsr2p0_zca1p0_zcd1p0"
.option rvc
.globl _start
_start:
  .rept 5
  tail callee
  .endr
  .space 4096
callee:
  ret
