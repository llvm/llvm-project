# REQUIRES: loongarch
# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=loongarch64 -mattr=+relax a.s -o a.64.o
# RUN: llvm-mc -filetype=obj -triple=loongarch64 -mattr=+relax c.s -o c.64.o
# RUN: ld.lld --relax -shared -soname=c.64.so c.64.o -o c.64.so

## Test the TLSDESC relaxation.
# RUN: ld.lld --relax -shared -z now a.64.o c.64.o -o a.64.so
# RUN: llvm-readobj -r -x .got a.64.so | FileCheck --check-prefix=GD64-RELA %s
# RUN: llvm-objdump --no-show-raw-insn -dr -h a.64.so | FileCheck %s --check-prefix=GD64

## FIXME: The transition from TLSDESC to IE/LE has not yet been implemented.
## Keep the dynamic relocations and hand them over to dynamic linker.

# RUN: ld.lld --relax -e 0 -z now a.64.o c.64.o -o a.64.le
# RUN: llvm-readobj -r -x .got a.64.le | FileCheck --check-prefix=LE64-RELA %s
# RUN: llvm-objdump --no-show-raw-insn -d -h a.64.le | FileCheck %s --check-prefix=LE64

# RUN: ld.lld --no-relax -e 0 -z now a.64.o c.64.o -o a.64.le.norelax
# RUN: llvm-objdump --no-show-raw-insn -d -h a.64.le.norelax | FileCheck %s --check-prefix=LE64-NORELAX

# RUN: ld.lld --relax -e 0 -z now a.64.o c.64.so -o a.64.ie
# RUN: llvm-readobj -r -x .got a.64.ie | FileCheck --check-prefix=IE64-RELA %s
# RUN: llvm-objdump --no-show-raw-insn -d -h a.64.ie | FileCheck %s --check-prefix=IE64

# RUN: ld.lld --no-relax -e 0 -z now a.64.o c.64.so -o a.64.ie.norelax
# RUN: llvm-objdump --no-show-raw-insn -d -h a.64.ie.norelax | FileCheck %s --check-prefix=IE64-NORELAX

# GD64-RELA:      .rela.dyn {
# GD64-RELA-NEXT:   0x20460 R_LARCH_TLS_DESC64 - 0x7FF
# GD64-RELA-NEXT:   0x20430 R_LARCH_TLS_DESC64 a 0x0
# GD64-RELA-NEXT:   0x20440 R_LARCH_TLS_DESC64 c 0x0
# GD64-RELA-NEXT:   0x20450 R_LARCH_TLS_DESC64 d 0x0
# GD64-RELA-NEXT: }
# GD64-RELA:      Hex dump of section '.got':
# GD64-RELA-NEXT: 0x00020430 00000000 00000000 00000000 00000000 .
# GD64-RELA-NEXT: 0x00020440 00000000 00000000 00000000 00000000 .
# GD64-RELA-NEXT: 0x00020450 00000000 00000000 00000000 00000000 .
# GD64-RELA-NEXT: 0x00020460 00000000 00000000 00000000 00000000 .

# GD64:   .got    00000040 0000000000020430

## &.got[a]-. = 0x20430 - 0x10318 = 16454<<2
# GD64:        10318: pcaddi  $a0, 16454
# GD64-NEXT:          ld.d    $ra, $a0, 0
# GD64-NEXT:          jirl    $ra, $ra, 0
# GD64-NEXT:          add.d   $a1, $a0, $tp

## &.got[b]-. = 0x20430+48 - 0x10328: 0x10 pages, page offset 0x460
## R_LARCH_RELAX does not appear in pairs. No relaxation.
# GD64:        10328: pcalau12i $a0, 16
# GD64-NEXT:          addi.d  $a0, $a0, 1120
# GD64-NEXT:          ld.d    $ra, $a0, 0
# GD64-NEXT:          jirl    $ra, $ra, 0
# GD64-NEXT:          add.d   $a2, $a0, $tp

## &.got[c]-. = 0x20430+16 - 0x1033c: 0x10 pages, page offset 0x440
## Without R_LARCH_RELAX relocation. No relaxation.
# GD64:        1033c: pcalau12i $a0, 16
# GD64-NEXT:          addi.d  $t0, $zero, 0
# GD64-NEXT:          addi.d  $a0, $a0, 1088
# GD64-NEXT:          addi.d  $t0, $t0, 1
# GD64-NEXT:          ld.d    $ra, $a0, 0
# GD64-NEXT:          addi.d  $t0, $t0, 1
# GD64-NEXT:          jirl    $ra, $ra, 0
# GD64-NEXT:          add.d   $a3, $a0, $tp

## &.got[d]-. = 0x20430+32 - 0x1035c = 16445<<2
# GD64:        1035c: pcaddi  $a0, 16445
# GD64-NEXT:          ld.d    $ra, $a0, 0
# GD64-NEXT:          jirl    $ra, $ra, 0
# GD64-NEXT:          add.d   $a4, $a0, $tp

# LE64-RELA:      .rela.dyn {
# LE64-RELA-NEXT:   0x30280 R_LARCH_TLS_DESC64 - 0x8
# LE64-RELA-NEXT:   0x30290 R_LARCH_TLS_DESC64 - 0x800
# LE64-RELA-NEXT:   0x302A0 R_LARCH_TLS_DESC64 - 0x1000
# LE64-RELA-NEXT:   0x302B0 R_LARCH_TLS_DESC64 - 0x7FF
# LE64-RELA-NEXT: }
# LE64-RELA:      Hex dump of section '.got':
# LE64-RELA-NEXT: 0x00030280 00000000 00000000 00000000 00000000 .
# LE64-RELA-NEXT: 0x00030290 00000000 00000000 00000000 00000000 .
# LE64-RELA-NEXT: 0x000302a0 00000000 00000000 00000000 00000000 .
# LE64-RELA-NEXT: 0x000302b0 00000000 00000000 00000000 00000000 .

# LE64:   .got    00000040 0000000000030280

## &.got[a]-. = 0x30280 - 0x20228 = 16406<<2
# LE64:        20228: pcaddi  $a0, 16406
# LE64-NEXT:          ld.d    $ra, $a0, 0
# LE64-NEXT:          jirl    $ra, $ra, 0
# LE64-NEXT:          add.d   $a1, $a0, $tp

## &.got[b]-. = 0x30280+48 - 0x20238: 0x10 pages, page offset 0x2b0
## R_LARCH_RELAX does not appear in pairs. No relaxation.
# LE64:        20238: pcalau12i $a0, 16
# LE64-NEXT:          addi.d  $a0, $a0, 688
# LE64-NEXT:          ld.d    $ra, $a0, 0
# LE64-NEXT:          jirl    $ra, $ra, 0
# LE64-NEXT:          add.d   $a2, $a0, $tp

## &.got[c]-. = 0x30280+16 - 0x2024c: 0x10 pages, page offset 0x290
## Without R_LARCH_RELAX relocation. No relaxation.
# LE64:        2024c: pcalau12i $a0, 16
# LE64-NEXT:          addi.d  $t0, $zero, 0
# LE64-NEXT:          addi.d  $a0, $a0, 656
# LE64-NEXT:          addi.d  $t0, $t0, 1
# LE64-NEXT:          ld.d    $ra, $a0, 0
# LE64-NEXT:          addi.d  $t0, $t0, 1
# LE64-NEXT:          jirl    $ra, $ra, 0
# LE64-NEXT:          add.d   $a3, $a0, $tp

## &.got[d]-. = 0x30280+32 - 0x2026c = 16397<<2
# LE64:        2026c: pcaddi  $a0, 16397
# LE64-NEXT:          ld.d    $ra, $a0, 0
# LE64-NEXT:          jirl    $ra, $ra, 0
# LE64-NEXT:          add.d   $a4, $a0, $tp

# LE64-NORELAX: .got    00000040 0000000000030288

## &.got[a]-. = 0x30288 - 0x20228 = 0x10 pages, page offset 0x288
# LE64-NORELAX:        20228: pcalau12i $a0, 16
# LE64-NORELAX-NEXT:          addi.d  $a0, $a0, 648
# LE64-NORELAX-NEXT:          ld.d    $ra, $a0, 0
# LE64-NORELAX-NEXT:          jirl    $ra, $ra, 0
# LE64-NORELAX-NEXT:          add.d   $a1, $a0, $tp

## &.got[b]-. = 0x30288+48 - 0x2023c: 0x10 pages, page offset 0x2b8
## R_LARCH_RELAX does not appear in pairs. No relaxation.
# LE64-NORELAX:        2023c: pcalau12i $a0, 16
# LE64-NORELAX-NEXT:          addi.d  $a0, $a0, 696
# LE64-NORELAX-NEXT:          ld.d    $ra, $a0, 0
# LE64-NORELAX-NEXT:          jirl    $ra, $ra, 0
# LE64-NORELAX-NEXT:          add.d   $a2, $a0, $tp

## &.got[c]-. = 0x30288+16 - 0x20250: 0x10 pages, page offset 0x298
## Without R_LARCH_RELAX relocation. No relaxation.
# LE64-NORELAX:        20250: pcalau12i $a0, 16
# LE64-NORELAX-NEXT:          addi.d  $t0, $zero, 0
# LE64-NORELAX-NEXT:          addi.d  $a0, $a0, 664
# LE64-NORELAX-NEXT:          addi.d  $t0, $t0, 1
# LE64-NORELAX-NEXT:          ld.d    $ra, $a0, 0
# LE64-NORELAX-NEXT:          addi.d  $t0, $t0, 1
# LE64-NORELAX-NEXT:          jirl    $ra, $ra, 0
# LE64-NORELAX-NEXT:          add.d   $a3, $a0, $tp

## &.got[d]-. = 0x30288+32 - 0x20270: 0x10 pages, page offset 0x2a8
# LE64-NORELAX:        20270: pcalau12i $a0, 16
# LE64-NORELAX-NEXT:          addi.d  $a0, $a0, 680
# LE64-NORELAX-NEXT:          ld.d    $ra, $a0, 0
# LE64-NORELAX-NEXT:          jirl    $ra, $ra, 0
# LE64-NORELAX-NEXT:          add.d   $a4, $a0, $tp

# IE64-RELA:      .rela.dyn {
# IE64-RELA-NEXT:   0x30430 R_LARCH_TLS_DESC64 - 0x8
# IE64-RELA-NEXT:   0x30460 R_LARCH_TLS_DESC64 - 0x7FF
# IE64-RELA-NEXT:   0x30440 R_LARCH_TLS_DESC64 c 0x0
# IE64-RELA-NEXT:   0x30450 R_LARCH_TLS_DESC64 d 0x0
# IE64-RELA-NEXT: }
# IE64-RELA:      Hex dump of section '.got':
# IE64-RELA-NEXT: 0x00030430 00000000 00000000 00000000 00000000 .
# IE64-RELA-NEXT: 0x00030440 00000000 00000000 00000000 00000000 .
# IE64-RELA-NEXT: 0x00030450 00000000 00000000 00000000 00000000 .
# IE64-RELA-NEXT: 0x00030460 00000000 00000000 00000000 00000000 .

# IE64:   .got           00000040 0000000000030430

## a and b are optimized to use LE. c and d are optimized to IE.
## &.got[a]-. = 0x30430 - 0x202f8 = 16462<<2
# IE64:        202f8: pcaddi  $a0, 16462
# IE64-NEXT:          ld.d    $ra, $a0, 0
# IE64-NEXT:          jirl    $ra, $ra, 0
# IE64-NEXT:          add.d   $a1, $a0, $tp

## &.got[b]-. = 0x30430+48 - 0x20308: 0x10 pages, page offset 0x460
## R_LARCH_RELAX does not appear in pairs. No relaxation.
# IE64:        20308: pcalau12i $a0, 16
# IE64-NEXT:          addi.d  $a0, $a0, 1120
# IE64-NEXT:          ld.d    $ra, $a0, 0
# IE64-NEXT:          jirl    $ra, $ra, 0
# IE64-NEXT:          add.d   $a2, $a0, $tp

## &.got[c]-. = 0x30430+16 - 0x2031c: 0x10 pages, page offset 0x440
## Without R_LARCH_RELAX relocation. No relaxation.
# IE64:        2031c: pcalau12i $a0, 16
# IE64-NEXT:          addi.d  $t0, $zero, 0
# IE64-NEXT:          addi.d  $a0, $a0, 1088
# IE64-NEXT:          addi.d  $t0, $t0, 1
# IE64-NEXT:          ld.d    $ra, $a0, 0
# IE64-NEXT:          addi.d  $t0, $t0, 1
# IE64-NEXT:          jirl    $ra, $ra, 0
# IE64-NEXT:          add.d   $a3, $a0, $tp

## &.got[d]-. = 0x30430+32 - 0x2033c = 16453<<2
# IE64:        2033c: pcaddi  $a0, 16453
# IE64-NEXT:          ld.d    $ra, $a0, 0
# IE64-NEXT:          jirl    $ra, $ra, 0
# IE64-NEXT:          add.d   $a4, $a0, $tp

# IE64-NORELAX: .got    00000040 0000000000030438

## &.got[a]-. = 0x30438 - 0x202f8 = 0x10 pages, page offset 0x438
# IE64-NORELAX:        202f8: pcalau12i $a0, 16
# IE64-NORELAX-NEXT:          addi.d  $a0, $a0, 1080
# IE64-NORELAX-NEXT:          ld.d    $ra, $a0, 0
# IE64-NORELAX-NEXT:          jirl    $ra, $ra, 0
# IE64-NORELAX-NEXT:          add.d   $a1, $a0, $tp

## &.got[b]-. = 0x30438+48 - 0x2030c: 0x10 pages, page offset 0x468
## R_LARCH_RELAX does not appear in pairs. No relaxation.
# IE64-NORELAX:        2030c: pcalau12i $a0, 16
# IE64-NORELAX-NEXT:          addi.d  $a0, $a0, 1128
# IE64-NORELAX-NEXT:          ld.d    $ra, $a0, 0
# IE64-NORELAX-NEXT:          jirl    $ra, $ra, 0
# IE64-NORELAX-NEXT:          add.d   $a2, $a0, $tp

## &.got[c]-. = 0x30438+16 - 0x20320: 0x10 pages, page offset 0x448
## Without R_LARCH_RELAX relocation. No relaxation.
# IE64-NORELAX:        20320: pcalau12i $a0, 16
# IE64-NORELAX-NEXT:          addi.d  $t0, $zero, 0
# IE64-NORELAX-NEXT:          addi.d  $a0, $a0, 1096
# IE64-NORELAX-NEXT:          addi.d  $t0, $t0, 1
# IE64-NORELAX-NEXT:          ld.d    $ra, $a0, 0
# IE64-NORELAX-NEXT:          addi.d  $t0, $t0, 1
# IE64-NORELAX-NEXT:          jirl    $ra, $ra, 0
# IE64-NORELAX-NEXT:          add.d   $a3, $a0, $tp

## &.got[d]-. = 0x30438+32 - 0x20340: 0x10 pages, page offset 0x458
# IE64-NORELAX:        20340: pcalau12i $a0, 16
# IE64-NORELAX-NEXT:          addi.d  $a0, $a0, 1112
# IE64-NORELAX-NEXT:          ld.d    $ra, $a0, 0
# IE64-NORELAX-NEXT:          jirl    $ra, $ra, 0
# IE64-NORELAX-NEXT:          add.d   $a4, $a0, $tp

#--- a.s
la.tls.desc $a0, a
add.d $a1, $a0, $tp

# ADDI.D does not have R_LARCH_RELAX. No relaxation.
pcalau12i $a0, %desc_pc_hi20(b)
.reloc .-4, R_LARCH_RELAX, 0
addi.d $a0, $a0, %desc_pc_lo12(b)
ld.d $ra, $a0, %desc_ld(b)
jirl $ra, $ra, %desc_call(b)
add.d $a2, $a0, $tp

# TLSDESC to LE. No relaxation.
pcalau12i $a0, %desc_pc_hi20(c)
addi.d $t0, $zero, 0
addi.d $a0, $a0, %desc_pc_lo12(c)
addi.d $t0, $t0, 1
ld.d $ra, $a0, %desc_ld(c)
addi.d $t0, $t0, 1
jirl $ra, $ra, %desc_call(c)
add.d $a3, $a0, $tp

# PCALAU12I and ADDI.D have R_LARCH_RELAX. We perform relaxation.
pcalau12i $a0, %desc_pc_hi20(d)
.reloc .-4, R_LARCH_RELAX, 0
addi.d $a0, $a0, %desc_pc_lo12(d)
.reloc .-4, R_LARCH_RELAX, 0
ld.d $ra, $a0, %desc_ld(d)
jirl $ra, $ra, %desc_call(d)
add.d $a4, $a0, $tp

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
