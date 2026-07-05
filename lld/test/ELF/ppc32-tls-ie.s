# REQUIRES: ppc
# RUN: llvm-mc -filetype=obj -triple=powerpc %s -o %t.o

# RUN: ld.lld -shared %t.o -o %t.so
# RUN: llvm-readobj -d -r %t.so | FileCheck --check-prefix=IE-REL %s
# RUN: llvm-objdump -d --no-show-raw-insn %t.so | FileCheck --check-prefix=IE %s

# RUN: ld.lld %t.o -o %t
# RUN: llvm-readelf -r %t | FileCheck --check-prefix=NOREL %s
# RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck --check-prefix=LE %s

# IE-REL:      FLAGS STATIC_TLS
## A non-preemptable symbol (b) has 0 st_shndx.
# IE-REL:      .rela.dyn {
# IE-REL-NEXT:   0x20258 R_PPC_TPREL32 - 0xC
# IE-REL-NEXT:   0x20254 R_PPC_TPREL32 a 0x0
# IE-REL-NEXT: }

## &.got[3] - _GLOBAL_OFFSET_TABLE_ = 12
# IE:      lwz 10, 12(9)
# IE-NEXT: add 10, 10, 2
## &.got[4] - _GLOBAL_OFFSET_TABLE_ = 16
# IE-NEXT: lwz 8, 16(7)
# IE-NEXT: lbzx 10, 8, 2

# NOREL: no relocations

## a@tprel = st_value(a)-0x7000 = -28664
## b@tprel = st_value(b)-0x7000 = -28660
# LE:      addis 10, 2, 0
# LE-NEXT: addi 10, 10, -28664
# LE-NEXT: addis 8, 2, 0
# LE-NEXT: lbz 10, -28660(8)

lwz 10, a@got@tprel(9)
add 10, 10, a@tls

lwz 8, c@got@tprel(7)
lbzx 10, 8, c@tls

## In IE, these instructions (op rT, rA, x@tls) are not changed.
# IE-NEXT: lhzx 12, 2, 2
# IE-NEXT: lwzx 13, 3, 2
# IE-NEXT: stbx 14, 4, 2
# IE-NEXT: sthx 15, 5, 2
# IE-NEXT: stwx 16, 6, 2
# IE-NEXT: lhax 17, 7, 2
# IE-NEXT: lwax 18, 8, 2
# IE-NEXT: lfsx 19, 9, 2
# IE-NEXT: lfdx 20, 10, 2
# IE-NEXT: stfsx 21, 11, 2
# IE-NEXT: stfdx 22, 12, 2

## In LE, these X-Form instructions are changed to their corresponding D-Form.
# LE-NEXT: lhz 12, -28660(2)
# LE-NEXT: lwz 13, -28660(3)
# LE-NEXT: stb 14, -28660(4)
# LE-NEXT: sth 15, -28660(5)
# LE-NEXT: stw 16, -28660(6)
# LE-NEXT: lha 17, -28660(7)
# LE-NEXT: lwa 18, -28660(8)
# LE-NEXT: lfs 19, -28660(9)
# LE-NEXT: lfd 20, -28660(10)
# LE-NEXT: stfs 21, -28660(11)
# LE-NEXT: stfd 22, -28660(12)

lhzx 12, 2, s@tls
lwzx 13, 3, i@tls
stbx 14, 4, c@tls
sthx 15, 5, s@tls
stwx 16, 6, i@tls
lhax 17, 7, s@tls
lwax 18, 8, i@tls
lfsx 19, 9, f@tls
lfdx 20, 10, d@tls
stfsx 21, 11, f@tls
stfdx 22, 12, d@tls
ldx 23, 13, l@tls
stdx 24, 14, l@tls

.section .tbss
.globl a
.zero 8
a:
.zero 4
c:
s:
i:
f:
d:
l:
