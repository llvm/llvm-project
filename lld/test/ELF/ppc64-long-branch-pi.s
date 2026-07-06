# REQUIRES: ppc
# RUN: llvm-mc -filetype=obj -triple=ppc64le %s -o %t.o
# RUN: echo 'SECTIONS { \
# RUN:       .text_low 0x2000: { *(.text_low) } \
# RUN:       .text_high 0x2002000 : { *(.text_high) } \
# RUN:       }' > %t.script
# RUN: ld.lld -pie -T %t.script %t.o -o %t
# RUN: llvm-readelf -S %t | FileCheck --check-prefix=SEC-PIE %s
# RUN: llvm-readobj -r %t | FileCheck --check-prefix=RELOC %s
# RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck %s

## RELATIVE relocs relocating NOBITS .branch_lt do not cause --check-dynamic-relocations errors.
# RUN: ld.lld -shared -T %t.script %t.o -o %t.so --apply-dynamic-relocs --check-dynamic-relocations
# RUN: llvm-readelf -S %t.so | FileCheck --check-prefix=SEC-SHARED %s
# RUN: llvm-objdump -d --no-show-raw-insn %t.so | FileCheck %s

# SEC-PIE:    Name       Type     Address          Off     Size   ES Flg Lk Inf Al
# SEC-PIE:    .got       PROGBITS 0000000002002120 2012120 000010 00  WA  0   0  8
# SEC-PIE:    .branch_lt NOBITS   0000000002002138 2012138 000020 00  WA  0   0  8

# SEC-SHARED: Name       Type     Address          Off     Size   ES Flg Lk Inf Al
# SEC-SHARED: .got       PROGBITS 0000000002002100 2012100 000010 00  WA  0   0  8
# SEC-SHARED: .branch_lt NOBITS   0000000002002118 2012118 000020 00  WA  0   0  8

# RELOC:      .rela.dyn {
# RELOC-NEXT:   0x2002128 R_PPC64_RELATIVE - 0x2002000
# RELOC-NEXT:   0x2002130 R_PPC64_RELATIVE - 0x2012120
# RELOC-NEXT:   0x2002138 R_PPC64_RELATIVE - 0x2002000
# RELOC-NEXT:   0x2002140 R_PPC64_RELATIVE - 0x2002008
# RELOC-NEXT:   0x2002148 R_PPC64_RELATIVE - 0x200200C
# RELOC-NEXT:   0x2002150 R_PPC64_RELATIVE - 0x2000
# RELOC-NEXT: }

# CHECK:      <_start>:
# CHECK-NEXT:     2000:       bl 0x2010
# CHECK-NEXT:                 bl 0x2002000
# CHECK-NEXT:                 bl 0x2030
# CHECK-NEXT:                 bl 0x2050

## &.branch_lt[0] - .TOC. = .branch_lt - (.got+0x8000) = -32744
# CHECK:      <__long_branch_>:
# CHECK-NEXT:     2010:       addis 12, 2, 0
# CHECK-NEXT:                 ld 12, -32744(12)
# CHECK-NEXT:                 mtctr 12
# CHECK-NEXT:                 bctr

## &.branch_lt[1] - .TOC. = .branch_lt - (.got+0x8000) = -32736
# CHECK:      <__long_branch_>:
# CHECK-NEXT:     2030:       addis 12, 2, 0
# CHECK-NEXT:                 ld 12, -32736(12)
# CHECK-NEXT:                 mtctr 12
# CHECK-NEXT:                 bctr

## &.branch_lt[2] - .TOC. = .branch_lt - (.got+0x8000) = -32728
# CHECK:      <__long_branch_>:
# CHECK-NEXT:     2050:       addis 12, 2, 0
# CHECK-NEXT:                 ld 12, -32728(12)
# CHECK-NEXT:                 mtctr 12
# CHECK-NEXT:                 bctr

.section .text_low, "ax", %progbits
.globl _start
_start:
bl .text_high     # Need a thunk
bl .text_high
bl .text_high+8   # Need a thunk
bl .text_high+0xc # Need a thunk

# CHECK:      <high_target>:
# CHECK-NEXT:  2002000:   bl 0x2004
# CHECK-NEXT:             bl 0x2004
# CHECK-NEXT:             bl 0x2002020

## &.branch_lt[3] - .TOC. = .branch_lt - (.got+0x8000) = -32720
# CHECK:      <__long_branch_>:
# CHECK-NEXT:  2002020:       addis 12, 2, 0
# CHECK-NEXT:                 ld 12, -32720(12)
# CHECK-NEXT:                 mtctr 12
# CHECK-NEXT:                 bctr

.section .text_high, "ax", %progbits
high_target:
bl .text_low+4
bl .text_low+4
bl .text_low      # Need a thunk
blr

## Force creation of .got
addis 3, 2, .text_high@got@ha

## The R_PPC64_RELATIVE makes sure .rela.dyn survives removeUnusedSyntheticSections.
.section .data
.quad .TOC.@tocbase
