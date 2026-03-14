# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=i386 %s -o %t.o
# RUN: ld.lld %t.o -o %t
# RUN: llvm-readelf -S -l %t | FileCheck --check-prefix=SEC %s
# RUN: llvm-objdump --no-print-imm-hex -d %t | FileCheck --check-prefix=DIS %s

# SEC: Name   Type     Address  Off    Size   ES Flg Lk Inf Al
# SEC: .tdata PROGBITS 00402200 000200 000001 00 WAT 0    0  1
# SEC: .tbss  NOBITS   00402300 000201 000008 00 WAT 0    0 256

# SEC: Type Offset   VirtAddr   PhysAddr   FileSiz MemSiz  Flg Align
# SEC: TLS  0x000200 0x00402200 0x00402200 0x00001 0x00108 R   0x100

## a@tprel = st_value(a) - p_memsz - (-p_vaddr-p_memsz & p_align-1)
## = 0 - 256 - 0 = -256
# DIS: leal -256(%ecx), %eax

lea a@ntpoff(%ecx), %eax

.section .tdata,"awT"
.byte 0

.section .tbss,"awT"
.p2align 8
a:
.quad 0
