# REQUIRES: riscv

# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+c,+zcmt,+relax zcmt.s -o zcmt.o
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+relax norvc.s -o norvc.o

# RUN: ld.lld -e _start --relax-tbljal zcmt.o -o enabled
# RUN: llvm-readelf -S -s enabled | FileCheck %s --check-prefix=HAS-JVT
# RUN: llvm-objdump -d --no-show-raw-insn -M no-aliases enabled \
# RUN:   | FileCheck %s --check-prefix=ZCMT

# RUN: ld.lld -e _start --no-relax-tbljal --relax-tbljal=zcmt zcmt.o -o explicit
# RUN: cmp enabled explicit
# RUN: not ld.lld --relax-tbljal=xqccmt zcmt.o 2>&1 | FileCheck %s --check-prefix=BAD-SCHEME
# BAD-SCHEME: error: unsupported table-jump scheme 'xqccmt' in --relax-tbljal= (expected zcmt)
# RUN: not ld.lld --relax-tbljal= zcmt.o 2>&1 | FileCheck %s --check-prefix=EMPTY
# EMPTY: error: unsupported table-jump scheme '' in --relax-tbljal= (expected zcmt)

# RUN: ld.lld -e _start zcmt.o -o default
# RUN: llvm-readelf -S -s default | FileCheck /dev/null \
# RUN:   --implicit-check-not=.riscv.jvt --implicit-check-not=__jvt_base$
# RUN: llvm-objdump -d --no-show-raw-insn -M no-aliases default \
# RUN:   | FileCheck %s --check-prefix=NOZCMT \
# RUN:     --implicit-check-not=cm.jt --implicit-check-not=cm.jalt

# RUN: ld.lld -e _start --relax-tbljal --no-relax-tbljal zcmt.o -o no-zcmt
# RUN: llvm-readelf -S -s no-zcmt | FileCheck /dev/null \
# RUN:   --implicit-check-not=.riscv.jvt --implicit-check-not=__jvt_base$
# RUN: llvm-objdump -d --no-show-raw-insn -M no-aliases no-zcmt \
# RUN:   | FileCheck %s --check-prefix=NOZCMT \
# RUN:     --implicit-check-not=cm.jt --implicit-check-not=cm.jalt

# RUN: ld.lld -e _start --no-relax --relax-tbljal zcmt.o -o no-relax 2>&1 \
# RUN:   | FileCheck %s --check-prefix=NO-RELAX-WARN
# RUN: llvm-readelf -S -s no-relax | FileCheck /dev/null \
# RUN:   --implicit-check-not=.riscv.jvt --implicit-check-not=__jvt_base$
# RUN: llvm-objdump -d --no-show-raw-insn -M no-aliases no-relax \
# RUN:   | FileCheck %s --check-prefix=NO-RELAX-DIS \
# RUN:     --implicit-check-not=cm.jt --implicit-check-not=cm.jalt

# RUN: ld.lld -e _start --relax-tbljal norvc.o -o norvc 2>&1 \
# RUN:   | FileCheck %s --check-prefix=NORVC-WARN
# RUN: llvm-readelf -S -s norvc | FileCheck /dev/null \
# RUN:   --implicit-check-not=.riscv.jvt --implicit-check-not=__jvt_base$
# RUN: llvm-objdump -d --no-show-raw-insn -M no-aliases norvc \
# RUN:   | FileCheck %s --check-prefix=NOZCMT \
# RUN:     --implicit-check-not=cm.jt --implicit-check-not=cm.jalt

# HAS-JVT: .riscv.jvt
# HAS-JVT: __jvt_base$

# ZCMT-LABEL: <_start>:
# ZCMT-COUNT-5: cm.jt 0x0
# ZCMT-NOT: cm.jt
# ZCMT-LABEL: <callee>:

# NOZCMT-LABEL: <_start>:
# NOZCMT-NEXT: jal zero, {{.*}} <callee>

# NO-RELAX-WARN: warning: --relax-tbljal requires --relax
# NO-RELAX-DIS-LABEL: <_start>:
# NO-RELAX-DIS-NEXT: auipc
# NO-RELAX-DIS-NEXT: jalr zero, {{.*}} <callee>

# NORVC-WARN: warning: --relax-tbljal is disabled because norvc.o is not marked RVC-capable

#--- zcmt.s
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

#--- norvc.s
.globl _start
_start:
  .rept 5
  tail callee
  .endr
  .space 4096
callee:
  ret
