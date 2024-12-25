# REQUIRES: loongarch
# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=loongarch64 a.s -o a.64.o
# RUN: llvm-mc -filetype=obj -triple=loongarch64 c.s -o c.64.o
# RUN: ld.lld -shared -soname=c.64.so c.64.o -o c.64.so

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

# GD64-RELA:      .rela.dyn {
# GD64-RELA-NEXT:   0x20568 R_LARCH_TLS_DESC64 - 0x1000
# GD64-RELA-NEXT:   0x20518 R_LARCH_TLS_DESC64 a 0x0
# GD64-RELA-NEXT:   0x20528 R_LARCH_TLS_DESC64 c 0x0
# GD64-RELA-NEXT:   0x20538 R_LARCH_TLS_DESC64 d 0x0
# GD64-RELA-NEXT:   0x20548 R_LARCH_TLS_DESC64 e 0x0
# GD64-RELA-NEXT:   0x20558 R_LARCH_TLS_DESC64 f 0x0
# GD64-RELA-NEXT: }
# GD64-RELA:      Hex dump of section '.got':
# GD64-RELA-NEXT: 0x00020518 00000000 00000000 00000000 00000000 .
# GD64-RELA-NEXT: 0x00020528 00000000 00000000 00000000 00000000 .
# GD64-RELA-NEXT: 0x00020538 00000000 00000000 00000000 00000000 .
# GD64-RELA-NEXT: 0x00020548 00000000 00000000 00000000 00000000 .
# GD64-RELA-NEXT: 0x00020558 00000000 00000000 00000000 00000000 .
# GD64-RELA-NEXT: 0x00020568 00000000 00000000 00000000 00000000 .

# GD64-REL:      .rel.dyn {
# GD64-REL-NEXT:   0x20538 R_LARCH_TLS_DESC64 -
# GD64-REL-NEXT:   0x204E8 R_LARCH_TLS_DESC64 a
# GD64-REL-NEXT:   0x204F8 R_LARCH_TLS_DESC64 c
# GD64-REL-NEXT:   0x20508 R_LARCH_TLS_DESC64 d
# GD64-REL-NEXT:   0x20518 R_LARCH_TLS_DESC64 e
# GD64-REL-NEXT:   0x20528 R_LARCH_TLS_DESC64 f
# GD64-REL-NEXT: }
# GD64-REL:      Hex dump of section '.got':
# GD64-REL-NEXT: 0x000204e8 00000000 00000000 00000000 00000000 .
# GD64-REL-NEXT: 0x000204f8 00000000 00000000 00000000 00000000 .
# GD64-REL-NEXT: 0x00020508 00000000 00000000 00000000 00000000 .
# GD64-REL-NEXT: 0x00020518 00000000 00000000 00000000 00000000 .
# GD64-REL-NEXT: 0x00020528 00000000 00000000 00000000 00000000 .
# GD64-REL-NEXT: 0x00020538 00000000 00000000 00100000 00000000 .

# GD64:      .got     00000060 0000000000020518

## &.got[a]-. = 0x20518 - 0x10398: 0x10 pages, page offset 0x518
# GD64:        10398: pcalau12i $a0, 16
# GD64-NEXT:          addi.d $t0, $zero, 1304
# GD64-NEXT:          lu32i.d	$t0, 0
# GD64-NEXT:          lu52i.d	$t0, $t0, 0
# GD64-NEXT:          add.d	$a0, $a0, $t0
# GD64-NEXT:          ld.d $ra, $a0, 0
# GD64-NEXT:          jirl $ra, $ra, 0
# GD64-NEXT:          add.d $a1, $a0, $tp

## &.got[b]-. = 0x20518+80 - 0x103b8: 0x10 pages, page offset 0x568
# GD64:        103b8: pcalau12i $a0, 16
# GD64-NEXT:          addi.d $t0, $zero, 1384
# GD64-NEXT:          lu32i.d	$t0, 0
# GD64-NEXT:          lu52i.d	$t0, $t0, 0
# GD64-NEXT:          add.d	$a0, $a0, $t0
# GD64-NEXT:          ld.d $ra, $a0, 0
# GD64-NEXT:          jirl $ra, $ra, 0
# GD64-NEXT:          add.d $a2, $a0, $tp

## &.got[c]-. = 0x20518+16 - 0x103d8: 0x10 pages, page offset 0x528
# GD64:        103d8: pcalau12i $a0, 16
# GD64-NEXT:          addi.d  $t0, $zero, 1320
# GD64-NEXT:          lu32i.d	$t0, 0
# GD64-NEXT:          lu52i.d	$t0, $t0, 0
# GD64-NEXT:          add.d	$a0, $a0, $t0
# GD64-NEXT:          ld.d $ra, $a0, 0
# GD64-NEXT:          jirl $ra, $ra, 0
# GD64-NEXT:          add.d $a3, $a0, $tp

## &.got[d]-. = 0x20518+32 - 0x103f8: 0x10 pages, page offset 0x538
# GD64:        103f8: pcalau12i $a0, 16
# GD64-NEXT:          addi.d  $t0, $zero, 1336
# GD64-NEXT:          lu32i.d	$t0, 0
# GD64-NEXT:          lu52i.d	$t0, $t0, 0
# GD64-NEXT:          add.d	$a0, $a0, $t0
# GD64-NEXT:          ld.d $ra, $a0, 0
# GD64-NEXT:          jirl $ra, $ra, 0
# GD64-NEXT:          add.d $a4, $a0, $tp

## &.got[e]-. = 0x20518+48 - 0x10418: 0x10 pages, page offset 0x548
# GD64:        10418: pcalau12i $a0, 16
# GD64-NEXT:          addi.d  $t0, $zero, 1352
# GD64-NEXT:          lu32i.d	$t0, 0
# GD64-NEXT:          lu52i.d	$t0, $t0, 0
# GD64-NEXT:          add.d	$a0, $a0, $t0
# GD64-NEXT:          ld.d $ra, $a0, 0
# GD64-NEXT:          jirl $ra, $ra, 0
# GD64-NEXT:          add.d $a5, $a0, $tp

## &.got[f]-. = 0x20518+64 - 0x10438: 0x10 pages, page offset 0x558
# GD64:        10438: pcalau12i $a0, 16
# GD64-NEXT:          addi.d  $t0, $zero, 1368
# GD64-NEXT:          lu32i.d	$t0, 0
# GD64-NEXT:          lu52i.d	$t0, $t0, 0
# GD64-NEXT:          add.d	$a0, $a0, $t0
# GD64-NEXT:          ld.d $ra, $a0, 0
# GD64-NEXT:          jirl $ra, $ra, 0
# GD64-NEXT:          add.d $a6, $a0, $tp

# NOREL: no relocations

# LE64-LABEL: <.text>:
## st_value(a) = 8
# LE64-NEXT:         nop
# LE64-NEXT:         ori     $a0, $zero, 8
# LE64-NEXT:         nop
# LE64-NEXT:         nop
# LE64-NEXT:         nop
# LE64-NEXT:         nop
# LE64-NEXT:         nop
# LE64-NEXT:         add.d   $a1, $a0, $tp
## st_value(b) = 0x1000
# LE64-NEXT:         lu12i.w $a0, 1
# LE64-NEXT:         ori     $a0, $a0, 0
# LE64-NEXT:         nop
# LE64-NEXT:         nop
# LE64-NEXT:         nop
# LE64-NEXT:         nop
# LE64-NEXT:         nop
# LE64-NEXT:         add.d   $a2, $a0, $tp
## st_value(c) = 0x7fffffff
# LE64-NEXT:         lu12i.w $a0, 524287
# LE64-NEXT:         ori	   $a0, $a0, 4095
# LE64-NEXT:         nop
# LE64-NEXT:         nop
# LE64-NEXT:         nop
# LE64-NEXT:         nop
# LE64-NEXT:         nop
# LE64-NEXT:         add.d   $a3, $a0, $tp
## st_value(d) = 0x8000,0000
# LE64-NEXT:         lu12i.w $a0, -524288
# LE64-NEXT:         ori	   $a0, $a0, 0
# LE64-NEXT:         lu32i.d $a0, 1
# LE64-NEXT:         lu52i.d $a0, $a0, 1
# LE64-NEXT:         nop
# LE64-NEXT:         nop
# LE64-NEXT:         nop
# LE64-NEXT:         add.d   $a4, $a0, $tp
## st_value(e) = 0x1,0000,0000
# LE64-NEXT:         nop
# LE64-NEXT:         ori	   $a0, $zero, 0
# LE64-NEXT:         lu32i.d $a0, 1
# LE64-NEXT:         nop
# LE64-NEXT:         nop
# LE64-NEXT:         nop
# LE64-NEXT:         nop
# LE64-NEXT:         add.d   $a5, $a0, $tp
## st_value(f) = 0x10,0000,0000,0000
# LE64-NEXT:         nop
# LE64-NEXT:         ori     $a0, $zero, 0
# LE64-NEXT:         nop
# LE64-NEXT:         lu52i.d $a0, $a0, 1
# LE64-NEXT:         nop
# LE64-NEXT:         nop
# LE64-NEXT:         nop
# LE64-NEXT:         add.d   $a6, $a0, $tp

# IE64-RELA:      .rela.dyn {
# IE64-RELA-NEXT:   0x304D8 R_LARCH_TLS_TPREL64 c 0x0
# IE64-RELA-NEXT:   0x304E0 R_LARCH_TLS_TPREL64 d 0x0
# IE64-RELA-NEXT:   0x304E8 R_LARCH_TLS_TPREL64 e 0x0
# IE64-RELA-NEXT:   0x304F0 R_LARCH_TLS_TPREL64 f 0x0
# IE64-RELA-NEXT: }
# IE64-RELA:      Hex dump of section '.got':
# IE64-RELA-NEXT: 0x000304d8 00000000 00000000 00000000 00000000 .
# IE64-RELA-NEXT: 0x000304e8 00000000 00000000 00000000 00000000 .

# IE64:       .got     00000020 00000000000304d8

## a and b are optimized to use LE. c, d, e and f are optimized to IE.
# IE64-LABEL: <.text>:
## st_value(a) = 8
# IE64-NEXT:         nop
# IE64-NEXT:         ori     $a0, $zero, 8
# IE64-NEXT:         nop
# IE64-NEXT:         nop
# IE64-NEXT:         nop
# IE64-NEXT:         nop
# IE64-NEXT:         nop
# IE64-NEXT:         add.d   $a1, $a0, $tp
## st_value(b) = 0x1000
# IE64-NEXT:         lu12i.w $a0, 1
# IE64-NEXT:         ori     $a0, $a0, 0
# IE64-NEXT:         nop
# IE64-NEXT:         nop
# IE64-NEXT:         nop
# IE64-NEXT:         nop
# IE64-NEXT:         nop
# IE64-NEXT:         add.d   $a2, $a0, $tp
## &.got[c]-. = 0x304d8 - 0x20378: 0x10 pages, page offset 0x4d8
# IE64-NEXT:  20378: pcalau12i $a0, 16
# IE64-NEXT:         addi.d    $t0, $zero, 1240
# IE64-NEXT:         lu32i.d	 $t0, 0
# IE64-NEXT:         lu52i.d	 $t0, $t0, 0
# IE64-NEXT:         ldx.d     $a0, $a0, $t0
# IE64-NEXT:         nop
# IE64-NEXT:         nop
# IE64-NEXT:         add.d   $a3, $a0, $tp
## &.got[d]-. = 0x304d8+8 - 0x20398: 0x10 pages, page offset 0x4e0
# IE64-NEXT:  20398: pcalau12i $a0, 16
# IE64-NEXT:         addi.d    $t0, $zero, 1248
# IE64-NEXT:         lu32i.d	 $t0, 0
# IE64-NEXT:         lu52i.d	 $t0, $t0, 0
# IE64-NEXT:         ldx.d     $a0, $a0, $t0
# IE64-NEXT:         nop
# IE64-NEXT:         nop
# IE64-NEXT:         add.d   $a4, $a0, $tp
## &.got[e]-. = 0x304d8+16 - 0x203b8: 0x10 pages, page offset 0x4e8
# IE64-NEXT:  203b8: pcalau12i $a0, 16
# IE64-NEXT:         addi.d    $t0, $zero, 1256
# IE64-NEXT:         lu32i.d	 $t0, 0
# IE64-NEXT:         lu52i.d	 $t0, $t0, 0
# IE64-NEXT:         ldx.d     $a0, $a0, $t0
# IE64-NEXT:         nop
# IE64-NEXT:         nop
# IE64-NEXT:         add.d   $a5, $a0, $tp
## &.got[f]-. = 0x304d8+32 - 0x203e8: 0x10 pages, page offset 0x4f0
# IE64-NEXT:  203d8: pcalau12i $a0, 16
# IE64-NEXT:         addi.d    $t0, $zero, 1264
# IE64-NEXT:         lu32i.d	 $t0, 0
# IE64-NEXT:         lu52i.d	 $t0, $t0, 0
# IE64-NEXT:         ldx.d     $a0, $a0, $t0
# IE64-NEXT:         nop
# IE64-NEXT:         nop
# IE64-NEXT:         add.d   $a6, $a0, $tp

#--- a.s
la.tls.desc $a0, $t0, a
add.d $a1, $a0, $tp

la.tls.desc $a0, $t0, b
add.d $a2, $a0, $tp

la.tls.desc $a0, $t0, c
add.d $a3, $a0, $tp

la.tls.desc $a0, $t0, d
add.d $a4, $a0, $tp

la.tls.desc $a0, $t0, e
add.d $a5, $a0, $tp

la.tls.desc $a0, $t0, f
add.d $a6, $a0, $tp

.section .tbss,"awT",@nobits
.globl a
.zero 8
a:
.zero 0x1000-8  ## Place b at 0x1000
b:

#--- c.s
.section .tbss,"awT",@nobits
.globl c, d, e, f
.zero 0x80000000-0x1000-1  ## Place c at 0x7fffffff
c:
.zero 1  ## Place d at 0x8000,0000
d:
.zero 0x100000000-0x80000000  ## Place e at 0x1,0000,0000
e:
.zero 0x10000000000000-0x100000000  ## Place f at 0x10,0000,0000,0000
f:

