# REQUIRES: loongarch
## Test that we can handle --emit-relocs while relaxing.

# RUN: llvm-mc --filetype=obj --triple=loongarch32 --mattr=+relax %s -o %t.32.o
# RUN: llvm-mc --filetype=obj --triple=loongarch64 --mattr=+relax %s -o %t.64.o
# RUN: ld.lld -Ttext=0x10000 --emit-relocs %t.32.o -o %t.32
# RUN: ld.lld -Ttext=0x10000 --emit-relocs %t.64.o -o %t.64
# RUN: llvm-objdump -dr %t.32 | FileCheck %s --check-prefix=LA32RELOC
# RUN: llvm-objdump -dr %t.64 | FileCheck %s --check-prefix=LA64RELOC

## -r should keep original relocations.
# RUN: ld.lld -r %t.64.o -o %t.64.r
# RUN: llvm-objdump -dr %t.64.r | FileCheck %s --check-prefix=RELAX

# RUN: ld.lld -Ttext=0x10000 --emit-relocs --no-relax %t.64.o -o %t.64.norelax
# RUN: llvm-objdump -dr %t.64.norelax | FileCheck %s --check-prefix=NORELAX

# LA32RELOC:      00010000 <_start>:
# LA32RELOC-NEXT:   pcalau12i $a0, 0
# LA32RELOC-NEXT:     R_LARCH_PCALA_HI20 _start
# LA32RELOC-NEXT:     R_LARCH_RELAX *ABS*
# LA32RELOC-NEXT:   addi.{{[dw]}} $a0, $a0, 0
# LA32RELOC-NEXT:     R_LARCH_PCALA_LO12 _start
# LA32RELOC-NEXT:     R_LARCH_RELAX *ABS*
# LA32RELOC-NEXT:   nop
# LA32RELOC-NEXT:     R_LARCH_ALIGN *ABS*+0xc
# LA32RELOC-NEXT:   nop
# LA32RELOC-NEXT:   ret

# LA64RELOC:      00010000 <_start>:
# LA64RELOC-NEXT:   pcaddi $a0, 0
# LA64RELOC-NEXT:     R_LARCH_RELAX _start
# LA64RELOC-NEXT:     R_LARCH_RELAX *ABS*
# LA64RELOC-NEXT:     R_LARCH_PCREL20_S2 _start
# LA64RELOC-NEXT:     R_LARCH_RELAX *ABS*
# LA64RELOC-NEXT:   nop
# LA64RELOC-NEXT:     R_LARCH_ALIGN *ABS*+0xc
# LA64RELOC-NEXT:   nop
# LA64RELOC-NEXT:   nop
# LA64RELOC-NEXT:   ret


# RELAX:      <_start>:
# RELAX-NEXT:   pcalau12i $a0, 0
# RELAX-NEXT:     R_LARCH_PCALA_HI20 _start
# RELAX-NEXT:     R_LARCH_RELAX *ABS*
# RELAX-NEXT:   addi.d $a0, $a0, 0
# RELAX-NEXT:     R_LARCH_PCALA_LO12 _start
# RELAX-NEXT:     R_LARCH_RELAX *ABS*
# RELAX-NEXT:   nop
# RELAX-NEXT:     R_LARCH_ALIGN *ABS*+0xc
# RELAX-NEXT:   nop
# RELAX-NEXT:   nop
# RELAX-NEXT:   ret

# NORELAX:      <_start>:
# NORELAX-NEXT:   pcalau12i $a0, 0
# NORELAX-NEXT:     R_LARCH_PCALA_HI20 _start
# NORELAX-NEXT:     R_LARCH_RELAX *ABS*
# NORELAX-NEXT:   addi.d $a0, $a0, 0
# NORELAX-NEXT:     R_LARCH_PCALA_LO12 _start
# NORELAX-NEXT:     R_LARCH_RELAX *ABS*
# NORELAX-NEXT:   nop
# NORELAX-NEXT:     R_LARCH_ALIGN *ABS*+0xc
# NORELAX-NEXT:   nop
# NORELAX-NEXT:   ret

.global _start
_start:
  la.pcrel $a0, _start
  .p2align 4
  ret
