# REQUIRES: ppc
# RUN: echo 'SECTIONS { \
# RUN:       .text_low 0x10010000: { *(.text_low) } \
# RUN:       .text_high 0x10080000 : { *(.text_high) } \
# RUN:       }' > %t.script

# RUN: llvm-mc -filetype=obj -triple=powerpc64le %s -o %t.o
# RUN: ld.lld -T %t.script --shared %t.o -o %t
# RUN: llvm-readelf -s %t | FileCheck %s --check-prefix=SYMBOL
# RUN: llvm-readelf -r %t | FileCheck %s --check-prefix=RELA
# RUN: llvm-objdump -d --no-show-raw-insn --mcpu=future %t | FileCheck %s

# RUN: llvm-mc -filetype=obj -triple=powerpc64 %s -o %t.o
# RUN: ld.lld -T %t.script --shared %t.o -o %t
# RUN: llvm-readelf -s %t | FileCheck %s --check-prefix=SYMBOL
# RUN: llvm-readelf -r %t | FileCheck %s --check-prefix=RELA
# RUN: llvm-objdump -d --no-show-raw-insn --mcpu=future %t | FileCheck %s

.text
.section .text_low, "ax", %progbits
# CHECK-LABEL: <GlobIntPCRel>:
# CHECK-NEXT:    pld 3, 458936(0), 1
# CHECK-NEXT:    lwa 3, 0(3)

# SYMBOL: Symbol table '.dynsym' contains 4 entries:
# SYMBOL:      00000000     0 NOTYPE  GLOBAL  DEFAULT     UND glob_int
# RELA:        100800b8  0000000100000014 R_PPC64_GLOB_DAT       0000000000000000 glob_int + 0
GlobIntPCRel:
  pld 3, glob_int@got@PCREL(0), 1
  lwa 3, 0(3)
  blr

# CHECK-LABEL: <GlobIntPCRelOffset>:
# CHECK-NEXT:    pld 3, 458928(0), 1
# CHECK-NEXT:    lwa 3, 8(3)
# SYMBOL:      00000000     0 NOTYPE  GLOBAL  DEFAULT     UND glob_int8
# RELA:        100800c0  0000000200000014 R_PPC64_GLOB_DAT       0000000000000000 glob_int8 + 0
GlobIntPCRelOffset:
  pld 3, glob_int8@got@PCREL(0), 1
  lwa 3, 8(3)
  blr

# CHECK-LABEL: <GlobIntPCRelBigOffset>:
# CHECK-NEXT:    pld 3, 200(0), 1
# CHECK-NEXT:    lwa 3, 64(3)
# SYMBOL:      00000000     0 NOTYPE  GLOBAL  DEFAULT     UND glob_int8_big
# RELA:        100800c8  0000000300000014 R_PPC64_GLOB_DAT       0000000000000000 glob_int8_big + 0

## Note that the first entry of the .got[0] should always be .TOC.
# SYMBOL: Symbol table '.symtab' contains 8 entries:
# SYMBOL: 1: 0000000010010000 0 NOTYPE LOCAL DEFAULT 6 GlobIntPCRel
# SYMBOL: 2: 0000000010010010 0 NOTYPE LOCAL DEFAULT 6 GlobIntPCRelOffset
# SYMBOL: 3: 0000000010080000 0 NOTYPE LOCAL DEFAULT 7 GlobIntPCRelBigOffset
.section .text_high, "ax", %progbits
GlobIntPCRelBigOffset:
  pld 3, glob_int8_big@got@PCREL(0), 1
  lwa 3, 64(3)
  blr
