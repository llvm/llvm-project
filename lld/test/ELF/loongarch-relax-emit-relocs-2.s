# REQUIRES: loongarch
## Test that we can handle --emit-relocs while relaxing.
## Call36 and tail36 need LA64 basic integer, so they donot have 32-bit version.

# RUN: llvm-mc --filetype=obj --triple=loongarch64 --mattr=+relax %s -o %t.64.o
# RUN: ld.lld -Ttext=0x10000 --emit-relocs %t.64.o -o %t.64
# RUN: llvm-objdump -dr %t.64 | FileCheck %s --check-prefix=RELAX

## -r should keep original relocations.
# RUN: ld.lld -r %t.64.o -o %t.64.r
# RUN: llvm-objdump -dr %t.64.r | FileCheck %s --check-prefix=CHECKR

## --no-relax should keep original relocations.
# RUN: ld.lld -Ttext=0x10000 --emit-relocs --no-relax %t.64.o -o %t.64.norelax
# RUN: llvm-objdump -dr %t.64.norelax | FileCheck %s --check-prefix=NORELAX

# RELAX:      00010000 <_start>:
# RELAX-NEXT:   bl  0
# RELAX-NEXT:     R_LARCH_B26 _start
# RELAX-NEXT:     R_LARCH_RELAX *ABS*
# RELAX-NEXT:   b   -4
# RELAX-NEXT:     R_LARCH_B26 _start
# RELAX-NEXT:     R_LARCH_RELAX *ABS*
# RELAX-NEXT:   nop
# RELAX-NEXT:     R_LARCH_ALIGN *ABS*+0xc
# RELAX-NEXT:   nop
# RELAX-NEXT:   ret

# CHECKR:      <_start>:
# CHECKR-NEXT:   pcaddu18i $ra, 0
# CHECKR-NEXT:     R_LARCH_CALL36 _start
# CHECKR-NEXT:     R_LARCH_RELAX *ABS*
# CHECKR-NEXT:   jirl   $ra, $ra, 0
# CHECKR-NEXT:   pcaddu18i $t0, 0
# CHECKR-NEXT:     R_LARCH_CALL36 _start
# CHECKR-NEXT:     R_LARCH_RELAX *ABS*
# CHECKR-NEXT:   jr     $t0
# CHECKR-NEXT:   nop
# CHECKR-NEXT:     R_LARCH_ALIGN *ABS*+0xc
# CHECKR-NEXT:   nop
# CHECKR-NEXT:   nop
# CHECKR-NEXT:   ret

# NORELAX:      <_start>:
# NORELAX-NEXT:   pcaddu18i $ra, 0
# NORELAX-NEXT:     R_LARCH_CALL36 _start
# NORELAX-NEXT:     R_LARCH_RELAX *ABS*
# NORELAX-NEXT:   jirl   $ra, $ra, 0
# NORELAX-NEXT:   pcaddu18i $t0, 0
# NORELAX-NEXT:     R_LARCH_CALL36 _start
# NORELAX-NEXT:     R_LARCH_RELAX *ABS*
# NORELAX-NEXT:   jirl $zero, $t0, -8
# NORELAX-NEXT:   ret
# NORELAX-NEXT:     R_LARCH_ALIGN *ABS*+0xc

.global _start
_start:
  call36 _start
  tail36 $t0, _start
  .p2align 4
  ret
