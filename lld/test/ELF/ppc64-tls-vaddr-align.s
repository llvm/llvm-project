# REQUIRES: ppc

# RUN: llvm-mc -filetype=obj -triple=powerpc64le %s -o %t.o
# RUN: ld.lld %t.o -o %t
# RUN: llvm-readelf -S -l %t | FileCheck --check-prefix=SEC %s
# RUN: llvm-objdump -d %t | FileCheck --check-prefix=DIS %s

# SEC: Name   Type     Address          Off    Size   ES Flg Lk Inf Al
# SEC: .tdata PROGBITS 0000000010020300 000300 000001 00 WAT 0    0  1
# SEC: .tbss  NOBITS   0000000010020400 000301 000008 00 WAT 0    0 256

# SEC: Type Offset   VirtAddr           PhysAddr           FileSiz  MemSiz   Flg Align
# SEC: TLS  0x000300 0x0000000010020300 0x0000000010020300 0x000001 0x000108 R   0x100

## We currently have a hack in Writer.cpp:fixSectionAlignments() to force
## p_vaddr(PT_TLS)%p_align(PT_TLS)=0, to work around bugs in some dynamic loaders.

## p_vaddr rounded down to p_align has TP offset -0x7000.
## The first address of PT_TLS (p_vaddr) has TP offset (p_vaddr%p_align - 0x7000).
## Once we delete the hack, it is likely p_vaddr%p_align != 0.

## a@tprel = st_value(a) + p_vaddr%p_align - 0x7000 = .tbss-.tdata + p_vaddr%p_align - 0x7000
## = 0x10020400-0x10020300 + 0 - 0x7000 = -28416
# DIS: ld 3, -28416(13)

ld 3, a@tprel(13)

.section .tdata,"awT"
.byte 0

.section .tbss,"awT"
.p2align 8
a:
.quad 0
