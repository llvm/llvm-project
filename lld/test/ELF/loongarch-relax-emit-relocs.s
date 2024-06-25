# REQUIRES: loongarch
## Test that we can handle --emit-relocs while relaxing.

# RUN: llvm-mc --filetype=obj --triple=loongarch32 --mattr=+relax %s -o %t.32.o
# RUN: llvm-mc --filetype=obj --triple=loongarch64 --mattr=+relax %s -o %t.64.o
# RUN: ld.lld -Ttext=0x10000 --emit-relocs %t.32.o -o %t.32
# RUN: ld.lld -Ttext=0x10000 --emit-relocs %t.64.o -o %t.64
# RUN: llvm-objdump -dr %t.32 | FileCheck %s
# RUN: llvm-objdump -dr %t.64 | FileCheck %s

## -r should keep original relocations.
# RUN: ld.lld -r %t.64.o -o %t.64.r
# RUN: llvm-objdump -dr %t.64.r | FileCheck %s --check-prefix=CHECKR

## --no-relax should keep original relocations.
## TODO Due to R_LARCH_RELAX is not relaxed, it plays same as --relax now.
# RUN: ld.lld -Ttext=0x10000 --emit-relocs --no-relax %t.64.o -o %t.64.norelax
# RUN: llvm-objdump -dr %t.64.norelax | FileCheck %s

# CHECK:      00010000 <_start>:
# CHECK-NEXT:   pcalau12i $a0, 0
# CHECK-NEXT:     R_LARCH_PCALA_HI20 _start
# CHECK-NEXT:     R_LARCH_RELAX *ABS*
# CHECK-NEXT:   addi.{{[dw]}} $a0, $a0, 0
# CHECK-NEXT:     R_LARCH_PCALA_LO12 _start
# CHECK-NEXT:     R_LARCH_RELAX *ABS*
# CHECK-NEXT:   nop
# CHECK-NEXT:     R_LARCH_ALIGN *ABS*+0xc
# CHECK-NEXT:   nop
# CHECK-NEXT:   ret

# CHECKR:      <_start>:
# CHECKR-NEXT:   pcalau12i $a0, 0
# CHECKR-NEXT:     R_LARCH_PCALA_HI20 _start
# CHECKR-NEXT:     R_LARCH_RELAX *ABS*
# CHECKR-NEXT:   addi.d $a0, $a0, 0
# CHECKR-NEXT:     R_LARCH_PCALA_LO12 _start
# CHECKR-NEXT:     R_LARCH_RELAX *ABS*
# CHECKR-NEXT:   nop
# CHECKR-NEXT:     R_LARCH_ALIGN *ABS*+0xc
# CHECKR-NEXT:   nop
# CHECKR-NEXT:   nop
# CHECKR-NEXT:   ret

.global _start
_start:
  la.pcrel $a0, _start
  .p2align 4
  ret
