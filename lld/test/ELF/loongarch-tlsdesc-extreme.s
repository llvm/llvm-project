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

## FIXME: The transition frome TLSDESC to IE/LE has not yet been implemented.
## Keep the dynamic relocations and hand them over to dynamic linker.

# RUN: ld.lld -e 0 -z now a.64.o c.64.o -o a.64.le
# RUN: llvm-readobj -r -x .got a.64.le | FileCheck --check-prefix=LE64-RELA %s

# RUN: ld.lld -e 0 -z now a.64.o c.64.so -o a.64.ie
# RUN: llvm-readobj -r -x .got a.64.ie | FileCheck --check-prefix=IE64-RELA %s

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

# LE64-RELA:      .rela.dyn {
# LE64-RELA-NEXT:   0x30318 R_LARCH_TLS_DESC64 - 0x8
# LE64-RELA-NEXT:   0x30328 R_LARCH_TLS_DESC64 - 0x7FFFFFFF
# LE64-RELA-NEXT:   0x30338 R_LARCH_TLS_DESC64 - 0x80000000
# LE64-RELA-NEXT:   0x30348 R_LARCH_TLS_DESC64 - 0x100000000
# LE64-RELA-NEXT:   0x30358 R_LARCH_TLS_DESC64 - 0x10000000000000
# LE64-RELA-NEXT:   0x30368 R_LARCH_TLS_DESC64 - 0x1000
# LE64-RELA-NEXT: }
# LE64-RELA:      Hex dump of section '.got':
# LE64-RELA-NEXT: 0x00030318 00000000 00000000 00000000 00000000 .
# LE64-RELA-NEXT: 0x00030328 00000000 00000000 00000000 00000000 .
# LE64-RELA-NEXT: 0x00030338 00000000 00000000 00000000 00000000 .
# LE64-RELA-NEXT: 0x00030348 00000000 00000000 00000000 00000000 .
# LE64-RELA-NEXT: 0x00030358 00000000 00000000 00000000 00000000 .
# LE64-RELA-NEXT: 0x00030368 00000000 00000000 00000000 00000000 .

# IE64-RELA:      .rela.dyn {
# IE64-RELA-NEXT:   0x30508 R_LARCH_TLS_DESC64 - 0x8
# IE64-RELA-NEXT:   0x30558 R_LARCH_TLS_DESC64 - 0x1000
# IE64-RELA-NEXT:   0x30518 R_LARCH_TLS_DESC64 c 0x0
# IE64-RELA-NEXT:   0x30528 R_LARCH_TLS_DESC64 d 0x0
# IE64-RELA-NEXT:   0x30538 R_LARCH_TLS_DESC64 e 0x0
# IE64-RELA-NEXT:   0x30548 R_LARCH_TLS_DESC64 f 0x0
# IE64-RELA-NEXT: }
# IE64-RELA:      Hex dump of section '.got':
# IE64-RELA-NEXT: 0x00030508 00000000 00000000 00000000 00000000 .
# IE64-RELA-NEXT: 0x00030518 00000000 00000000 00000000 00000000 .
# IE64-RELA-NEXT: 0x00030528 00000000 00000000 00000000 00000000 .
# IE64-RELA-NEXT: 0x00030538 00000000 00000000 00000000 00000000 .
# IE64-RELA-NEXT: 0x00030548 00000000 00000000 00000000 00000000 .
# IE64-RELA-NEXT: 0x00030558 00000000 00000000 00000000 00000000 .

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

