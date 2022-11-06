# REQUIRES: aarch64

# RUN: llvm-mc -filetype=obj -triple=aarch64 %s -o %t.o
# RUN: ld.lld %t.o -o %t
# RUN: llvm-readelf -S -l %t | FileCheck --check-prefix=SEC %s
# RUN: llvm-objdump --no-print-imm-hex -d %t | FileCheck --check-prefix=DIS %s

# SEC: Name   Type     Address          Off    Size   ES Flg Lk Inf Al
# SEC: .tdata PROGBITS 0000000000220200 000200 000001 00 WAT 0    0  1
# SEC: .tbss  NOBITS   0000000000220300 000201 000008 00 WAT 0    0 256

# SEC: Type Offset   VirtAddr           PhysAddr           FileSiz  MemSiz   Flg Align
# SEC: TLS  0x000200 0x0000000000220200 0x0000000000220200 0x000001 0x000108 R   0x100

## We currently have a hack in Writer.cpp:fixSectionAlignments() to force
## p_vaddr(PT_TLS)%p_align(PT_TLS)=0, to work around bugs in some dynamic loaders.

## a@tprel = st_value(a) + GAP + (p_vaddr-GAP_ABOVE_TP & p_align-1) =
## .tbss-.tdata + 16 + GAP_ABOVE_TP + (p_vaddr-GAP_ABOVE_TP & p_align-1) =
## 0x220300-0x220200 + 16 + (0x220200-16 & 0x100-1) = 512
# DIS: add x0, x0, #512

add x0, x0, :tprel_lo12_nc:a

.section .tdata,"awT"
.byte 0

.section .tbss,"awT"
.p2align 8
a:
.quad 0
