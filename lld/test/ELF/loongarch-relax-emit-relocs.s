# REQUIRES: loongarch
## Test that we can handle --emit-relocs while relaxing.

# RUN: llvm-mc --filetype=obj --triple=loongarch32 --mattr=+relax %s -o %t.32.o
# RUN: llvm-mc --filetype=obj --triple=loongarch64 --mattr=+relax --defsym ELF64=1 %s -o %t.64.o
# RUN: ld.lld -Ttext=0x10000 -section-start=.got=0x20000 --emit-relocs %t.32.o -o %t.32
# RUN: ld.lld -Ttext=0x10000 -section-start=.got=0x20000 --emit-relocs %t.64.o -o %t.64
# RUN: llvm-objdump -dr %t.32 | FileCheck %s --check-prefixes=RELAX,RELAX32
# RUN: llvm-objdump -dr %t.64 | FileCheck %s --check-prefixes=RELAX,RELAX64

## -r should keep original relocations.
# RUN: ld.lld -r %t.64.o -o %t.64.r
# RUN: llvm-objdump -dr %t.64.r | FileCheck %s --check-prefix=CHECKR

## --no-relax should keep original relocations.
# RUN: ld.lld -Ttext=0x10000 -section-start=.got=0x20000 --emit-relocs --no-relax %t.64.o -o %t.64.norelax
# RUN: llvm-objdump -dr %t.64.norelax | FileCheck %s --check-prefix=NORELAX

# RELAX:      00010000 <_start>:
# RELAX-NEXT:   pcaddi $a0, 0
# RELAX-NEXT:     R_LARCH_RELAX _start
# RELAX-NEXT:     R_LARCH_RELAX *ABS*
# RELAX-NEXT:     R_LARCH_PCREL20_S2 _start
# RELAX-NEXT:     R_LARCH_RELAX *ABS*
# RELAX-NEXT:   pcaddi $a0, -1
# RELAX-NEXT:     R_LARCH_RELAX _start
# RELAX-NEXT:     R_LARCH_RELAX *ABS*
# RELAX-NEXT:     R_LARCH_PCREL20_S2 _start
# RELAX-NEXT:     R_LARCH_RELAX *ABS*

# RELAX64-NEXT:  bl  -8
# RELAX64-NEXT:    R_LARCH_B26 _start
# RELAX64-NEXT:    R_LARCH_RELAX *ABS*
# RELAX64-NEXT:  b   -12
# RELAX64-NEXT:    R_LARCH_B26 _start
# RELAX64-NEXT:    R_LARCH_RELAX *ABS*

# RELAX-NEXT:   lu12i.w   $a0, 0
# RELAX-NEXT:     R_LARCH_TLS_LE_HI20 a
# RELAX-NEXT:   ori       $a0, $a0, 0
# RELAX-NEXT:     R_LARCH_TLS_LE_LO12 a
# RELAX-NEXT:   pcaddi    $a0, [[#]]
# RELAX-NEXT:     R_LARCH_RELAX a
# RELAX-NEXT:     R_LARCH_RELAX *ABS*
# RELAX-NEXT:     R_LARCH_TLS_GD_PCREL20_S2 a
# RELAX-NEXT:     R_LARCH_RELAX *ABS*
# RELAX-NEXT:   pcaddi    $a0, [[#]]
# RELAX-NEXT:     R_LARCH_RELAX a
# RELAX-NEXT:     R_LARCH_RELAX *ABS*
# RELAX-NEXT:     R_LARCH_TLS_LD_PCREL20_S2 a
# RELAX-NEXT:     R_LARCH_RELAX *ABS*
# RELAX-NEXT:   addi.{{[dw]}} $a0, $tp, 0
# RELAX-NEXT:     R_LARCH_RELAX a
# RELAX-NEXT:     R_LARCH_RELAX *ABS*
# RELAX-NEXT:     R_LARCH_RELAX a
# RELAX-NEXT:     R_LARCH_RELAX *ABS*
# RELAX-NEXT:     R_LARCH_TLS_LE_LO12_R a
# RELAX-NEXT:     R_LARCH_RELAX *ABS*

# RELAX32-NEXT:  nop
# RELAX32-NEXT:    R_LARCH_ALIGN *ABS*+0xc
# RELAX32-NEXT:  ret

# RELAX64-NEXT:  nop
# RELAX64-NEXT:    R_LARCH_ALIGN *ABS*+0xc
# RELAX64-NEXT:  nop
# RELAX64-NEXT:  nop
# RELAX64-NEXT:  ret

# NORELAX:      <_start>:
# NORELAX-NEXT:   pcalau12i $a0, 0
# NORELAX-NEXT:     R_LARCH_PCALA_HI20 _start
# NORELAX-NEXT:     R_LARCH_RELAX *ABS*
# NORELAX-NEXT:   addi.d    $a0, $a0, 0
# NORELAX-NEXT:     R_LARCH_PCALA_LO12 _start
# NORELAX-NEXT:     R_LARCH_RELAX *ABS*
# NORELAX-NEXT:   pcalau12i $a0, 16
# NORELAX-NEXT:     R_LARCH_GOT_PC_HI20 _start
# NORELAX-NEXT:     R_LARCH_RELAX *ABS*
# NORELAX-NEXT:   ld.d      $a0, $a0, 0
# NORELAX-NEXT:     R_LARCH_GOT_PC_LO12 _start
# NORELAX-NEXT:     R_LARCH_RELAX *ABS*
# NORELAX-NEXT:   pcaddu18i $ra, 0
# NORELAX-NEXT:     R_LARCH_CALL36 _start
# NORELAX-NEXT:     R_LARCH_RELAX *ABS*
# NORELAX-NEXT:   jirl   $ra, $ra, -16
# NORELAX-NEXT:   pcaddu18i $a0, 0
# NORELAX-NEXT:     R_LARCH_CALL36 _start
# NORELAX-NEXT:     R_LARCH_RELAX *ABS*
# NORELAX-NEXT:   jirl $zero, $a0, -24
# NORELAX-NEXT:   lu12i.w   $a0, 0
# NORELAX-NEXT:     R_LARCH_TLS_LE_HI20 a
# NORELAX-NEXT:   ori       $a0, $a0, 0
# NORELAX-NEXT:     R_LARCH_TLS_LE_LO12 a
# NORELAX-NEXT:   pcalau12i $a0, 16
# NORELAX-NEXT:     R_LARCH_TLS_GD_PC_HI20 a
# NORELAX-NEXT:     R_LARCH_RELAX *ABS*
# NORELAX-NEXT:   addi.d    $a0, $a0, 8
# NORELAX-NEXT:     R_LARCH_GOT_PC_LO12 a
# NORELAX-NEXT:     R_LARCH_RELAX *ABS*
# NORELAX-NEXT:   pcalau12i $a0, 16
# NORELAX-NEXT:     R_LARCH_TLS_LD_PC_HI20 a
# NORELAX-NEXT:     R_LARCH_RELAX *ABS*
# NORELAX-NEXT:   addi.d    $a0, $a0, 8
# NORELAX-NEXT:     R_LARCH_GOT_PC_LO12 a
# NORELAX-NEXT:     R_LARCH_RELAX *ABS*
# NORELAX-NEXT:   lu12i.w   $a0, 0
# NORELAX-NEXT:     R_LARCH_TLS_LE_HI20_R a
# NORELAX-NEXT:     R_LARCH_RELAX *ABS*
# NORELAX-NEXT:   add.d     $a0, $a0, $tp
# NORELAX-NEXT:     R_LARCH_TLS_LE_ADD_R a
# NORELAX-NEXT:     R_LARCH_RELAX *ABS*
# NORELAX-NEXT:   addi.d    $a0, $a0, 0
# NORELAX-NEXT:     R_LARCH_TLS_LE_LO12_R a
# NORELAX-NEXT:     R_LARCH_RELAX *ABS*
# NORELAX-NEXT:   nop
# NORELAX-NEXT:     R_LARCH_ALIGN *ABS*+0xc
# NORELAX-NEXT:   nop
# NORELAX-NEXT:   nop
# NORELAX-NEXT:   ret

# CHECKR:      <_start>:
# CHECKR-NEXT:   pcalau12i $a0, 0
# CHECKR-NEXT:     R_LARCH_PCALA_HI20 _start
# CHECKR-NEXT:     R_LARCH_RELAX *ABS*
# CHECKR-NEXT:   addi.d    $a0, $a0, 0
# CHECKR-NEXT:     R_LARCH_PCALA_LO12 _start
# CHECKR-NEXT:     R_LARCH_RELAX *ABS*
# CHECKR-NEXT:   pcalau12i $a0, 0
# CHECKR-NEXT:     R_LARCH_GOT_PC_HI20 _start
# CHECKR-NEXT:     R_LARCH_RELAX *ABS*
# CHECKR-NEXT:   ld.d      $a0, $a0, 0
# CHECKR-NEXT:     R_LARCH_GOT_PC_LO12 _start
# CHECKR-NEXT:     R_LARCH_RELAX *ABS*
# CHECKR-NEXT:   pcaddu18i $ra, 0
# CHECKR-NEXT:     R_LARCH_CALL36 _start
# CHECKR-NEXT:     R_LARCH_RELAX *ABS*
# CHECKR-NEXT:   jirl   $ra, $ra, 0
# CHECKR-NEXT:   pcaddu18i $a0, 0
# CHECKR-NEXT:     R_LARCH_CALL36 _start
# CHECKR-NEXT:     R_LARCH_RELAX *ABS*
# CHECKR-NEXT:   jr     $a0
# CHECKR-NEXT:   lu12i.w   $a0, 0
# CHECKR-NEXT:     R_LARCH_TLS_LE_HI20 a
# CHECKR-NEXT:   ori       $a0, $a0, 0
# CHECKR-NEXT:     R_LARCH_TLS_LE_LO12 a
# CHECKR-NEXT:   pcalau12i $a0, 0
# CHECKR-NEXT:     R_LARCH_TLS_GD_PC_HI20 a
# CHECKR-NEXT:     R_LARCH_RELAX *ABS*
# CHECKR-NEXT:   addi.d    $a0, $a0, 0
# CHECKR-NEXT:     R_LARCH_GOT_PC_LO12 a
# CHECKR-NEXT:     R_LARCH_RELAX *ABS*
# CHECKR-NEXT:   pcalau12i $a0, 0
# CHECKR-NEXT:     R_LARCH_TLS_LD_PC_HI20 a
# CHECKR-NEXT:     R_LARCH_RELAX *ABS*
# CHECKR-NEXT:   addi.d    $a0, $a0, 0
# CHECKR-NEXT:     R_LARCH_GOT_PC_LO12 a
# CHECKR-NEXT:     R_LARCH_RELAX *ABS*
# CHECKR-NEXT:   lu12i.w   $a0, 0
# CHECKR-NEXT:     R_LARCH_TLS_LE_HI20_R a
# CHECKR-NEXT:     R_LARCH_RELAX *ABS*
# CHECKR-NEXT:   add.d     $a0, $a0, $tp
# CHECKR-NEXT:     R_LARCH_TLS_LE_ADD_R a
# CHECKR-NEXT:     R_LARCH_RELAX *ABS*
# CHECKR-NEXT:   addi.d    $a0, $a0, 0
# CHECKR-NEXT:     R_LARCH_TLS_LE_LO12_R a
# CHECKR-NEXT:     R_LARCH_RELAX *ABS*
# CHECKR-NEXT:   nop
# CHECKR-NEXT:     R_LARCH_ALIGN *ABS*+0xc
# CHECKR-NEXT:   nop
# CHECKR-NEXT:   nop
# CHECKR-NEXT:   ret

.macro add dst, src1, src2, src3
.ifdef ELF64
  add.d \dst, \src1, \src2, \src3
.else
  add.w \dst, \src1, \src2, \src3
.endif
.endm

.macro addi dst, src1, src2
.ifdef ELF64
  addi.d \dst, \src1, \src2
.else
  addi.w \dst, \src1, \src2
.endif
.endm

.global _start
_start:
  la.pcrel $a0, _start
  la.got   $a0, _start

.ifdef ELF64
  call36 _start
  tail36 $a0, _start
.endif

  la.tls.le $a0, a  # without R_LARCH_RELAX reloaction
  la.tls.gd $a0, a
  la.tls.ld $a0, a

  lu12i.w $a0, %le_hi20_r(a)
  add $a0, $a0, $tp, %le_add_r(a)
  addi $a0, $a0, %le_lo12_r(a)

  .p2align 4
  ret

.section .tbss,"awT",@nobits
.globl a
a:
.zero 4
