# REQUIRES: loongarch
## Test that --emit-relocs plus --no-relax produces correct offsets
## when R_LARCH_ALIGN is in effect.

# RUN: llvm-mc --filetype=obj --triple=loongarch32 --mattr=+relax %s -o %t.32.o
# RUN: llvm-mc --filetype=obj --triple=loongarch64 --mattr=+relax --defsym ELF64=1 %s -o %t.64.o
# RUN: ld.lld -Ttext=0x10000 --emit-relocs --no-relax %t.32.o -o %t.32
# RUN: ld.lld -Ttext=0x10000 --emit-relocs --no-relax %t.64.o -o %t.64
# RUN: llvm-objdump -dr %t.32 | FileCheck %s --check-prefixes=CHECK32
# RUN: llvm-objdump -dr %t.64 | FileCheck %s --check-prefixes=CHECK64

.text
.globl _start
_start:
  .reloc ., R_LARCH_ALIGN, 0x4
  nop
  la.pcrel $a0, _start
.ifdef ELF64
  la.pcrel $a1, $t0, _start
.endif
  nop
## For catching any trailing relocs
  nop
.size _start, .-_start

# CHECK32:      00010000 <_start>:
# CHECK32-NEXT:     pcaddu12i $a0, 0
# CHECK32-NEXT:         R_LARCH_ALIGN *ABS*+0x4
# CHECK32-NEXT:         R_LARCH_PCADD_HI20 _start
# CHECK32-NEXT:         R_LARCH_RELAX *ABS*
# CHECK32-NEXT:     addi.w $a0, $a0, 0
# CHECK32-NEXT:         R_LARCH_PCADD_LO12
# CHECK32-NEXT:         R_LARCH_RELAX *ABS*
# CHECK32-NEXT:     nop
# CHECK32-NEXT:     nop

# CHECK64:      00010000 <_start>:
# CHECK64-NEXT:     pcalau12i $a0, 0
# CHECK64-NEXT:         R_LARCH_ALIGN *ABS*+0x4
# CHECK64-NEXT:         R_LARCH_PCALA_HI20 _start
# CHECK64-NEXT:         R_LARCH_RELAX *ABS*
# CHECK64-NEXT:     addi.d $a0, $a0, 0
# CHECK64-NEXT:         R_LARCH_PCALA_LO12 _start
# CHECK64-NEXT:         R_LARCH_RELAX *ABS*
# CHECK64-NEXT:     pcalau12i $a1, 0
# CHECK64-NEXT:         R_LARCH_PCALA_HI20 _start
# CHECK64-NEXT:     addi.d $t0, $zero, 0
# CHECK64-NEXT:         R_LARCH_PCALA_LO12 _start
# CHECK64-NEXT:     lu32i.d $t0, 0
# CHECK64-NEXT:         R_LARCH_PCALA64_LO20 _start
# CHECK64-NEXT:     lu52i.d $t0, $t0, 0
# CHECK64-NEXT:         R_LARCH_PCALA64_HI12 _start
# CHECK64-NEXT:     add.d $a1, $a1, $t0
# CHECK64-NEXT:     nop
# CHECK64-NEXT:     nop
