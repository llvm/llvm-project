# REQUIRES: riscv

# RUN: rm -rf %t && split-file %s %t && cd %t

# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+c,+relax noattr.s -o noattr.o
# RUN: ld.lld -e _start --relax-tbljal noattr.o -o noattr
# RUN: llvm-objdump -d --no-show-raw-insn -M no-aliases noattr | FileCheck %s --check-prefix=JUMP
# RUN: llvm-readelf -S -s noattr | FileCheck %s --check-prefix=HAS-JVT
# RUN: llvm-readelf -A noattr | FileCheck %s --check-prefix=ATTR

# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+c,+relax noattr-nocandidate.s -o noattr-nocandidate.o
# RUN: ld.lld -e _start --relax-tbljal noattr-nocandidate.o -o noattr-nocandidate
# RUN: llvm-readelf -h -S -l -s -A noattr-nocandidate | FileCheck %s \
# RUN:   --check-prefix=NO-ATTR --implicit-check-not=.riscv.attributes \
# RUN:   --implicit-check-not=RISCV_ATTRIBUTES --implicit-check-not=__jvt_base$ \
# RUN:   --implicit-check-not="TagName: arch"

# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+c,+relax nozcmt.s -o nozcmt.o
# RUN: ld.lld -e _start --relax-tbljal nozcmt.o -o nozcmt
# RUN: llvm-objdump -d --no-show-raw-insn -M no-aliases nozcmt | FileCheck %s --check-prefix=JUMP
# RUN: llvm-readelf -S -s nozcmt | FileCheck %s --check-prefix=HAS-JVT
# RUN: llvm-readelf -A nozcmt | FileCheck %s --check-prefix=ATTR

# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+c,+zcmt,+relax candidate.s -o candidate.o
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+c,+relax support-nozcmt.s -o support-nozcmt.o
# RUN: ld.lld -e _start --relax-tbljal candidate.o support-nozcmt.o -o mixed
# RUN: llvm-objdump -d --no-show-raw-insn -M no-aliases mixed | FileCheck %s --check-prefix=JUMP
# RUN: llvm-readelf -S -s mixed | FileCheck %s --check-prefix=HAS-JVT
# RUN: llvm-readelf -A mixed | FileCheck %s --check-prefix=ATTR

# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+c,+relax compatible-nocandidate.s -o compatible-nocandidate.o
# RUN: ld.lld -e _start --relax-tbljal compatible-nocandidate.o -o compatible-nocandidate
# RUN: llvm-readelf -S -s compatible-nocandidate | FileCheck /dev/null \
# RUN:   --implicit-check-not=.riscv.jvt --implicit-check-not=__jvt_base$

# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+c,+zcmt,+relax zcd.s -o zcd.o
# RUN: ld.lld -e _start --relax-tbljal zcd.o -o zcd 2>&1 | FileCheck %s --check-prefix=ZCD
# RUN: llvm-readelf -S -s zcd | FileCheck /dev/null \
# RUN:   --implicit-check-not=.riscv.jvt --implicit-check-not=__jvt_base$

## Only the first code range uses the JVT. Preserve its M extension, the
## incompatible Zcd range, the intervening data and the untouched M range.
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+c,+relax regions.s -o regions.o
# RUN: ld.lld -e _start --relax-tbljal regions.o -o regions
# RUN: llvm-readelf -sW regions | FileCheck %s --check-prefix=REGION-SYM
# RUN: llvm-objdump -d --no-show-raw-insn -M no-aliases regions | FileCheck %s --check-prefix=REGION-DIS

# REGION-SYM:      NOTYPE LOCAL DEFAULT {{[0-9]+}} $xrv64i2p1_m2p0_{{.*}}_zcmt1p0
# REGION-SYM:      NOTYPE LOCAL DEFAULT {{[0-9]+}} $xrv64{{.*}}_zcd1p0{{$}}
# REGION-SYM:      NOTYPE LOCAL DEFAULT {{[0-9]+}} $d{{$}}
# REGION-SYM:      NOTYPE LOCAL DEFAULT {{[0-9]+}} $xrv64i2p1_m2p0_c2p0_zmmul1p0_zca1p0{{$}}
# REGION-DIS-LABEL: <_start>:
# REGION-DIS-COUNT-5: cm.jt
# REGION-DIS-LABEL: <incompatible>:
# REGION-DIS-NEXT: jal zero,
# REGION-DIS-LABEL: <callee>:
# REGION-DIS-NEXT: mul a0, a0, a1

# JUMP: cm.jt
# HAS-JVT: .riscv.jvt
# HAS-JVT: __jvt_base$
# ATTR: TagName: arch
# ATTR-NEXT: Value:
# ATTR-SAME: zicsr
# ATTR-SAME: zca
# ATTR-SAME: zcmt
# NO-ATTR: ELF Header:
# ZCD: warning: --relax-tbljal is disabled because
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

#--- regions.s
.option arch, +m
.globl _start
_start:
  .rept 5
  tail callee
  .endr
.option push
.option arch, +d
incompatible:
  tail callee
.option pop
  .space 4096
callee:
  mul a0, a0, a1
  ret
