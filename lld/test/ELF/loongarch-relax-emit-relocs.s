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
# RELAX32-NEXT:  nop
# RELAX32-NEXT:    R_LARCH_ALIGN *ABS*+0xc
# RELAX32-NEXT:  nop
# RELAX32-NEXT:  ret

# RELAX64-NEXT:  bl  -8
# RELAX64-NEXT:    R_LARCH_B26 _start
# RELAX64-NEXT:    R_LARCH_RELAX *ABS*
# RELAX64-NEXT:  b   -12
# RELAX64-NEXT:    R_LARCH_B26 _start
# RELAX64-NEXT:    R_LARCH_RELAX *ABS*
# RELAX64-NEXT:  ret
# RELAX64-NEXT:    R_LARCH_ALIGN *ABS*+0xc

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
# NORELAX-NEXT:   ret
# NORELAX-NEXT:     R_LARCH_ALIGN *ABS*+0xc

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
# CHECKR-NEXT:   nop
# CHECKR-NEXT:     R_LARCH_ALIGN *ABS*+0xc
# CHECKR-NEXT:   nop
# CHECKR-NEXT:   nop
# CHECKR-NEXT:   ret

.global _start
_start:
  la.pcrel $a0, _start
  la.got   $a0, _start

.ifdef ELF64
  call36 _start
  tail36 $a0, _start
.endif
  .p2align 4
  ret
