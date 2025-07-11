# REQUIRES: loongarch
# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=loongarch64 -mattr=+relax a.s -o a.64.o
# RUN: llvm-mc -filetype=obj -triple=loongarch64 -mattr=+relax c.s -o c.64.o
# RUN: ld.lld --relax -shared -soname=c.64.so c.64.o -o c.64.so

## Test the TLSDESC relaxation.
# RUN: ld.lld --relax -shared -z now a.64.o c.64.o -o a.64.so
# RUN: llvm-readobj -r -x .got a.64.so | FileCheck --check-prefix=GD64-RELA %s
# RUN: llvm-objdump --no-show-raw-insn -dr -h a.64.so | FileCheck %s --check-prefix=GD64

## FIXME: IE/LE relaxation have not yet been implemented, --relax/--no-relax obtain the same results.
## Transition from TLSDESC to IE/LE. Also check --emit-relocs.
# RUN: ld.lld -e 0 -z now --emit-relocs a.64.o c.64.o -o a.64.le
# RUN: llvm-readobj -r -x .got a.64.le 2>&1 | FileCheck --check-prefix=LE64-RELA %s
# RUN: llvm-objdump --no-show-raw-insn -dr -h a.64.le | FileCheck %s --check-prefix=LE64

# RUN: ld.lld --no-relax -e 0 -z now a.64.o c.64.o -o a.64.le.norelax
# RUN: llvm-objdump --no-show-raw-insn -d -h a.64.le.norelax | FileCheck %s --check-prefix=LE64-NORELAX

# RUN: ld.lld --relax -e 0 -z now --emit-relocs a.64.o c.64.so -o a.64.ie
# RUN: llvm-readobj -r -x .got a.64.ie | FileCheck --check-prefix=IE64-RELA %s
# RUN: llvm-objdump --no-show-raw-insn -dr -h a.64.ie | FileCheck %s --check-prefix=IE64

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

# LE64-RELA: could not find section '.got'

## a@tprel = 0x8
# LE64:        20158: nop
# LE64-NEXT:            R_LARCH_TLS_DESC_PC_HI20 a
# LE64-NEXT:            R_LARCH_RELAX *ABS*
# LE64-NEXT:          nop
# LE64-NEXT:            R_LARCH_TLS_DESC_PC_LO12 a
# LE64-NEXT:            R_LARCH_RELAX *ABS*
# LE64-NEXT:          nop
# LE64-NEXT:            R_LARCH_TLS_DESC_LD a
# LE64-NEXT:            R_LARCH_RELAX *ABS*
# LE64-NEXT:          ori     $a0, $zero, 8
# LE64-NEXT:            R_LARCH_TLS_DESC_CALL a
# LE64-NEXT:            R_LARCH_RELAX *ABS*
# LE64-NEXT:          add.d   $a1, $a0, $tp

## b@tprel = 0x7ff
# LE64:        2016c: nop
# LE64-NEXT:            R_LARCH_TLS_DESC_PC_HI20 b
# LE64-NEXT:            R_LARCH_RELAX *ABS*
# LE64-NEXT:          nop
# LE64-NEXT:            R_LARCH_TLS_DESC_PC_LO12 b
# LE64-NEXT:          nop
# LE64-NEXT:            R_LARCH_TLS_DESC_LD b
# LE64-NEXT:          ori     $a0, $zero, 2047
# LE64-NEXT:            R_LARCH_TLS_DESC_CALL b
# LE64-NEXT:          add.d   $a2, $a0, $tp

## c@tprel = 0x800
## Without R_LARCH_RELAX relocation. No relaxation.
# LE64:        20180: nop
# LE64-NEXT:            R_LARCH_TLS_DESC_PC_HI20 c
# LE64-NEXT:          addi.d  $t0, $zero, 0
# LE64-NEXT:          nop
# LE64-NEXT:            R_LARCH_TLS_DESC_PC_LO12 c
# LE64-NEXT:          addi.d  $t0, $t0, 1
# LE64-NEXT:          nop
# LE64-NEXT:            R_LARCH_TLS_DESC_LD c
# LE64-NEXT:          addi.d  $t0, $t0, 1
# LE64-NEXT:          ori     $a0, $zero, 2048
# LE64-NEXT:            R_LARCH_TLS_DESC_CALL c
# LE64-NEXT:          add.d   $a3, $a0, $tp

## d@tprel = 0x1000
# LE64:        201a0: nop
# LE64-NEXT:            R_LARCH_TLS_DESC_PC_HI20 d
# LE64-NEXT:            R_LARCH_RELAX *ABS*
# LE64-NEXT:          nop
# LE64-NEXT:            R_LARCH_TLS_DESC_PC_LO12 d
# LE64-NEXT:            R_LARCH_RELAX *ABS*
# LE64-NEXT:          lu12i.w $a0, 1
# LE64-NEXT:            R_LARCH_TLS_DESC_LD d
# LE64-NEXT:          ori     $a0, $a0, 0
# LE64-NEXT:            R_LARCH_TLS_DESC_CALL d
# LE64-NEXT:          add.d   $a4, $a0, $tp

## a@tprel = 0x8
# LE64-NORELAX:        20158: nop
# LE64-NORELAX-NEXT:          nop
# LE64-NORELAX-NEXT:          nop
# LE64-NORELAX-NEXT:          ori     $a0, $zero, 8
# LE64-NORELAX-NEXT:          add.d   $a1, $a0, $tp

## b@tprel = 0x7ff
# LE64-NORELAX:        2016c: nop
# LE64-NORELAX-NEXT:          nop
# LE64-NORELAX-NEXT:          nop
# LE64-NORELAX-NEXT:          ori     $a0, $zero, 2047
# LE64-NORELAX-NEXT:          add.d   $a2, $a0, $tp

## c@tprel = 0x800
## Without R_LARCH_RELAX relocation. No relaxation.
# LE64-NORELAX:        20180: nop
# LE64-NORELAX-NEXT:          addi.d  $t0, $zero, 0
# LE64-NORELAX-NEXT:          nop
# LE64-NORELAX-NEXT:          addi.d  $t0, $t0, 1
# LE64-NORELAX-NEXT:          nop
# LE64-NORELAX-NEXT:          addi.d  $t0, $t0, 1
# LE64-NORELAX-NEXT:          ori     $a0, $zero, 2048
# LE64-NORELAX-NEXT:          add.d   $a3, $a0, $tp

## d@tprel = 0x1000
# LE64-NORELAX:        201a0: nop
# LE64-NORELAX-NEXT:          nop
# LE64-NORELAX-NEXT:          lu12i.w $a0, 1
# LE64-NORELAX-NEXT:          ori     $a0, $a0, 0
# LE64-NORELAX-NEXT:          add.d   $a4, $a0, $tp

# IE64-RELA:      .rela.dyn {
# IE64-RELA-NEXT:   0x30408 R_LARCH_TLS_TPREL64 c 0x0
# IE64-RELA-NEXT:   0x30410 R_LARCH_TLS_TPREL64 d 0x0
# IE64-RELA-NEXT: }
# IE64-RELA:      Hex dump of section '.got':
# IE64-RELA-NEXT: 0x00030408 00000000 00000000 00000000 00000000 .

# IE64:   .got           00000010 0000000000030408

## a and b are optimized to use LE. c and d are optimized to IE.
## a@tprel = 0x8
# IE64:        202c8: nop
# IE64-NEXT:            R_LARCH_TLS_DESC_PC_HI20 a
# IE64-NEXT:            R_LARCH_RELAX *ABS*
# IE64-NEXT:          nop
# IE64-NEXT:            R_LARCH_TLS_DESC_PC_LO12 a
# IE64-NEXT:            R_LARCH_RELAX *ABS*
# IE64-NEXT:          nop
# IE64-NEXT:            R_LARCH_TLS_DESC_LD a
# IE64-NEXT:            R_LARCH_RELAX *ABS*
# IE64-NEXT:          ori     $a0, $zero, 8
# IE64-NEXT:            R_LARCH_TLS_DESC_CALL a
# IE64-NEXT:            R_LARCH_RELAX *ABS*
# IE64-NEXT:          add.d   $a1, $a0, $tp

## b@tprel = 0x7ff
# IE64:        202dc: nop
# IE64-NEXT:            R_LARCH_TLS_DESC_PC_HI20 b
# IE64-NEXT:            R_LARCH_RELAX *ABS*
# IE64-NEXT:          nop
# IE64-NEXT:            R_LARCH_TLS_DESC_PC_LO12 b
# IE64-NEXT:          nop
# IE64-NEXT:            R_LARCH_TLS_DESC_LD b
# IE64-NEXT:          ori     $a0, $zero, 2047
# IE64-NEXT:            R_LARCH_TLS_DESC_CALL b
# IE64-NEXT:          add.d   $a2, $a0, $tp

## &.got[c]-. = 0x30408 - 0x20300: 0x10 pages, page offset 0x408
## Without R_LARCH_RELAX relocation. No relaxation.
# IE64:        202f0: nop
# IE64-NEXT:            R_LARCH_TLS_DESC_PC_HI20 c
# IE64-NEXT:          addi.d  $t0, $zero, 0
# IE64-NEXT:          nop
# IE64-NEXT:            R_LARCH_TLS_DESC_PC_LO12 c
# IE64-NEXT:          addi.d  $t0, $t0, 1
# IE64-NEXT:          pcalau12i $a0, 16
# IE64-NEXT:            R_LARCH_TLS_DESC_LD c
# IE64-NEXT:          addi.d  $t0, $t0, 1
# IE64-NEXT:          ld.d    $a0, $a0, 1032
# IE64-NEXT:            R_LARCH_TLS_DESC_CALL c
# IE64-NEXT:          add.d   $a3, $a0, $tp

## &.got[d]-. = 0x30408+8 - 0x20318: 0x10 pages, page offset 0x410
# IE64:        20310: nop
# IE64-NEXT:            R_LARCH_TLS_DESC_PC_HI20 d
# IE64-NEXT:            R_LARCH_RELAX *ABS*
# IE64-NEXT:          nop
# IE64-NEXT:            R_LARCH_TLS_DESC_PC_LO12 d
# IE64-NEXT:            R_LARCH_RELAX *ABS*
# IE64-NEXT:          pcalau12i $a0, 16
# IE64-NEXT:            R_LARCH_TLS_DESC_LD d
# IE64-NEXT:          ld.d    $a0, $a0, 1040
# IE64-NEXT:            R_LARCH_TLS_DESC_CALL d
# IE64-NEXT:          add.d   $a4, $a0, $tp

# IE64-NORELAX: .got    00000010 0000000000030408

## a@tprel = 0x8
# IE64-NORELAX:        202c8: nop
# IE64-NORELAX-NEXT:          nop
# IE64-NORELAX-NEXT:          nop
# IE64-NORELAX-NEXT:          ori     $a0, $zero, 8
# IE64-NORELAX-NEXT:          add.d   $a1, $a0, $tp

## b@tprel = 0x7ff
# IE64-NORELAX:        202dc: nop
# IE64-NORELAX-NEXT:          nop
# IE64-NORELAX-NEXT:          nop
# IE64-NORELAX-NEXT:          ori     $a0, $zero, 2047
# IE64-NORELAX-NEXT:          add.d   $a2, $a0, $tp

## &.got[c]-. = 0x30408 - 0x20300: 0x10 pages, page offset 0x408
## Without R_LARCH_RELAX relocation. No relaxation.
# IE64-NORELAX:        202f0: nop
# IE64-NORELAX-NEXT:          addi.d  $t0, $zero, 0
# IE64-NORELAX-NEXT:          nop
# IE64-NORELAX-NEXT:          addi.d  $t0, $t0, 1
# IE64-NORELAX-NEXT:          pcalau12i $a0, 16
# IE64-NORELAX-NEXT:          addi.d  $t0, $t0, 1
# IE64-NORELAX-NEXT:          ld.d    $a0, $a0, 1032
# IE64-NORELAX-NEXT:          add.d   $a3, $a0, $tp

## &.got[d]-. = 0x30408+8 - 0x20318: 0x10 pages, page offset 0x410
# IE64-NORELAX:        20310: nop
# IE64-NORELAX-NEXT:          nop
# IE64-NORELAX-NEXT:          pcalau12i $a0, 16
# IE64-NORELAX-NEXT:          ld.d    $a0, $a0, 1040
# IE64-NORELAX-NEXT:          add.d   $a4, $a0, $tp

#--- a.s
la.tls.desc $a0, a
add.d $a1, $a0, $tp

# ADDI.D does not have R_LARCH_RELAX. No relaxation when it is not optimized to IE/LE (--shared).
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
