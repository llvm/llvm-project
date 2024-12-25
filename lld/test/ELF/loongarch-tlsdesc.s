# REQUIRES: loongarch
# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=loongarch64 a.s -o a.64.o
# RUN: llvm-mc -filetype=obj -triple=loongarch64 c.s -o c.64.o
# RUN: ld.lld -shared -soname=c.64.so c.64.o -o c.64.so
# RUN: llvm-mc -filetype=obj -triple=loongarch32 --defsym ELF32=1 a.s -o a.32.o
# RUN: llvm-mc -filetype=obj -triple=loongarch32 --defsym ELF32=1 c.s -o c.32.o
# RUN: ld.lld -shared -soname=c.32.so c.32.o -o c.32.so

# RUN: ld.lld -shared -z now a.64.o c.64.o -o a.64.so
# RUN: llvm-readobj -r -x .got a.64.so | FileCheck --check-prefix=GD64-RELA %s
# RUN: llvm-objdump --no-show-raw-insn -h -d a.64.so | FileCheck %s --check-prefix=GD64

# RUN: ld.lld -shared -z now a.64.o c.64.o -o rel.64.so -z rel
# RUN: llvm-readobj -r -x .got rel.64.so | FileCheck --check-prefix=GD64-REL %s

## Transition from TLSDESC to IE/LE.

# RUN: ld.lld -e 0 -z now a.64.o c.64.o -o a.64.le
# RUN: llvm-readelf -r a.64.le | FileCheck --check-prefix=NOREL %s
# RUN: llvm-objdump --no-show-raw-insn -h -d a.64.le | FileCheck %s --check-prefix=LE64

# RUN: ld.lld -e 0 -z now a.64.o c.64.so -o a.64.ie
# RUN: llvm-readobj -r -x .got a.64.ie | FileCheck --check-prefix=IE64-RELA %s
# RUN: llvm-objdump --no-show-raw-insn -h -d a.64.ie | FileCheck %s --check-prefix=IE64

## 32-bit code is mostly the same. We only test a few variants.

# RUN: ld.lld -shared -z now a.32.o c.32.o -o rel.32.so -z rel
# RUN: llvm-readobj -r -x .got rel.32.so | FileCheck --check-prefix=GD32-REL %s

# GD64-RELA:      .rela.dyn {
# GD64-RELA-NEXT:   0x20458 R_LARCH_TLS_DESC64 - 0x7FF
# GD64-RELA-NEXT:   0x20428 R_LARCH_TLS_DESC64 a 0x0
# GD64-RELA-NEXT:   0x20438 R_LARCH_TLS_DESC64 c 0x0
# GD64-RELA-NEXT:   0x20448 R_LARCH_TLS_DESC64 d 0x0
# GD64-RELA-NEXT: }
# GD64-RELA:      Hex dump of section '.got':
# GD64-RELA-NEXT: 0x00020428 00000000 00000000 00000000 00000000 .
# GD64-RELA-NEXT: 0x00020438 00000000 00000000 00000000 00000000 .
# GD64-RELA-NEXT: 0x00020448 00000000 00000000 00000000 00000000 .
# GD64-RELA-NEXT: 0x00020458 00000000 00000000 00000000 00000000 .

# GD64-REL:      .rel.dyn {
# GD64-REL-NEXT:   0x20438 R_LARCH_TLS_DESC64 -
# GD64-REL-NEXT:   0x20408 R_LARCH_TLS_DESC64 a
# GD64-REL-NEXT:   0x20418 R_LARCH_TLS_DESC64 c
# GD64-REL-NEXT:   0x20428 R_LARCH_TLS_DESC64 d
# GD64-REL-NEXT: }
# GD64-REL:      Hex dump of section '.got':
# GD64-REL-NEXT: 0x00020408 00000000 00000000 00000000 00000000 .
# GD64-REL-NEXT: 0x00020418 00000000 00000000 00000000 00000000 .
# GD64-REL-NEXT: 0x00020428 00000000 00000000 00000000 00000000 .
# GD64-REL-NEXT: 0x00020438 00000000 00000000 ff070000 00000000 .

# GD64:      .got     00000040 0000000000020428

## &.got[a]-. = 0x20428 - 0x10318: 0x10 pages, page offset 0x428
# GD64:        10318: pcalau12i $a0, 16
# GD64-NEXT:          addi.d $a0, $a0, 1064
# GD64-NEXT:          ld.d $ra, $a0, 0
# GD64-NEXT:          jirl $ra, $ra, 0
# GD64-NEXT:          add.d $a1, $a0, $tp

## &.got[b]-. = 0x20428+48 - 0x1032c: 0x10 pages, page offset 0x458
# GD64:        1032c: pcalau12i $a0, 16
# GD64-NEXT:          addi.d $a0, $a0, 1112
# GD64-NEXT:          ld.d $ra, $a0, 0
# GD64-NEXT:          jirl $ra, $ra, 0
# GD64-NEXT:          add.d $a2, $a0, $tp

## &.got[c]-. = 0x20428+16 - 0x10340: 0x10 pages, page offset 0x438
# GD64:        10340: pcalau12i $a0, 16
# GD64-NEXT:          addi.d $a0, $a0, 1080
# GD64-NEXT:          ld.d $ra, $a0, 0
# GD64-NEXT:          jirl $ra, $ra, 0
# GD64-NEXT:          add.d $a3, $a0, $tp

## &.got[d]-. = 0x20428+32 - 0x10354: 0x10 pages, page offset 0x448
# GD64:        10354: pcalau12i $a0, 16
# GD64-NEXT:          addi.d $a0, $a0, 1096
# GD64-NEXT:          ld.d $ra, $a0, 0
# GD64-NEXT:          jirl $ra, $ra, 0
# GD64-NEXT:          add.d $a4, $a0, $tp

# NOREL: no relocations

# LE64-LABEL: <.text>:
## st_value(a) = 8
# LE64-NEXT:         nop
# LE64-NEXT:         nop
# LE64-NEXT:         nop
# LE64-NEXT:         ori     $a0, $zero, 8
# LE64-NEXT:         add.d   $a1, $a0, $tp
## st_value(b) = 2047
# LE64-NEXT:         nop
# LE64-NEXT:         nop
# LE64-NEXT:         nop
# LE64-NEXT:         ori     $a0, $zero, 2047
# LE64-NEXT:         add.d   $a2, $a0, $tp
## st_value(c) = 2048
# LE64-NEXT:         nop
# LE64-NEXT:         nop
# LE64-NEXT:         nop
# LE64-NEXT:         ori     $a0, $zero, 2048
# LE64-NEXT:         add.d   $a3, $a0, $tp
## st_value(d) = 4096
# LE64-NEXT:         nop
# LE64-NEXT:         nop
# LE64-NEXT:         lu12i.w $a0, 1
# LE64-NEXT:         ori     $a0, $a0, 0
# LE64-NEXT:         add.d   $a4, $a0, $tp

# IE64-RELA:      .rela.dyn {
# IE64-RELA-NEXT:   0x303F8 R_LARCH_TLS_TPREL64 c 0x0
# IE64-RELA-NEXT:   0x30400 R_LARCH_TLS_TPREL64 d 0x0
# IE64-RELA-NEXT: }
# IE64-RELA:      Hex dump of section '.got':
# IE64-RELA-NEXT: 0x000303f8 00000000 00000000 00000000 00000000 .

# IE64:       .got     00000010 00000000000303f8

## a and b are optimized to use LE. c and d are optimized to IE.
# IE64-LABEL: <.text>:
# IE64-NEXT:         nop
# IE64-NEXT:         nop
# IE64-NEXT:         nop
# IE64-NEXT:         ori     $a0, $zero, 8
# IE64-NEXT:         add.d   $a1, $a0, $tp
# IE64-NEXT:         nop
# IE64-NEXT:         nop
# IE64-NEXT:         nop
# IE64-NEXT:         ori     $a0, $zero, 2047
# IE64-NEXT:         add.d   $a2, $a0, $tp
## &.got[c]-. = 0x303f8 - 0x202f8: 0x10 pages, page offset 0x3f8
# IE64-NEXT:         nop
# IE64-NEXT:         nop
# IE64-NEXT:  202f8: pcalau12i $a0, 16
# IE64-NEXT:         ld.d      $a0, $a0, 1016
# IE64-NEXT:         add.d   $a3, $a0, $tp
## &.got[d]-. = 0x303f8+8 - 0x2030c: 0x10 pages, page offset 0x400
# IE64-NEXT:         nop
# IE64-NEXT:         nop
# IE64-NEXT:  2030c: pcalau12i $a0, 16
# IE64-NEXT:         ld.d      $a0, $a0, 1024
# IE64-NEXT:         add.d   $a4, $a0, $tp

# GD32-REL:      .rel.dyn {
# GD32-REL-NEXT:    0x202B4 R_LARCH_TLS_DESC32 -
# GD32-REL-NEXT:    0x2029C R_LARCH_TLS_DESC32 a
# GD32-REL-NEXT:    0x202A4 R_LARCH_TLS_DESC32 c
# GD32-REL-NEXT:    0x202AC R_LARCH_TLS_DESC32 d
# GD32-REL-NEXT: }
# GD32-REL:      Hex dump of section '.got':
# GD32-REL-NEXT: 0x0002029c 00000000 00000000 00000000 00000000 .
# GD32-REL-NEXT: 0x000202ac 00000000 00000000 00000000 ff070000 .

#--- a.s
.macro add dst, src1, src2
.ifdef ELF32
add.w \dst, \src1, \src2
.else
add.d \dst, \src1, \src2
.endif
.endm

la.tls.desc $a0, a
add $a1, $a0, $tp

la.tls.desc $a0, b
add $a2, $a0, $tp

la.tls.desc $a0, c
add $a3, $a0, $tp

la.tls.desc $a0, d
add $a4, $a0, $tp

.section .tbss,"awT",@nobits
.globl a
.zero 8
a:
.zero 2039  ## Place b at 0x7ff
b:
.zero 1

#--- c.s
.section .tbss,"awT",@nobits
.globl c, d
c:
.zero 2048  ## Place d at 0x1000
d:
.zero 4
