# REQUIRES: ppc

# RUN: llvm-mc -filetype=obj -triple=powerpc64le-unknown-linux %s -o %t.o
# RUN: ld.lld --no-toc-optimize %t.o -o %t
# RUN: llvm-readelf -x .rodata -x .R_PPC64_TOC -x .eh_frame %t | FileCheck %s --check-prefix=DATALE
# RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck %s

# RUN: llvm-mc -filetype=obj -triple=powerpc64-unknown-linux %s -o %t.o
# RUN: ld.lld --no-toc-optimize %t.o -o %t
# RUN: llvm-readelf -x .rodata -x .R_PPC64_TOC -x .eh_frame %t | FileCheck %s --check-prefix=DATABE
# RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck %s

.text
.global _start
_start:
.Lfoo:
	li      0,1
	li      3,42
	sc

.section .toc,"aw",@progbits
.L1:
  .quad 22, 37, 89, 47

.section .R_PPC64_TOC16_LO_DS,"ax",@progbits
  ld 1, .L1@toc@l(2)

# CHECK-LABEL: Disassembly of section .R_PPC64_TOC16_LO_DS:
# CHECK: ld 1, -32760(2)

.section .R_PPC64_TOC16_LO,"ax",@progbits
  addi  1, 2, .L1@toc@l

# CHECK-LABEL: Disassembly of section .R_PPC64_TOC16_LO:
# CHECK: addi 1, 2, -32760

.section .R_PPC64_TOC16_HI,"ax",@progbits
  addis 1, 2, .L1@toc@h

# CHECK-LABEL: Disassembly of section .R_PPC64_TOC16_HI:
# CHECK: addis 1, 2, -1

.section .R_PPC64_TOC,"a",@progbits
  .quad .TOC.@tocbase

# SEC: .got PROGBITS 0000000010020208

## tocbase = .got+0x8000 = 0x10028208
# DATALE-LABEL: section '.R_PPC64_TOC':
# DATALE-NEXT:    e8810210 00000000

# DATABE-LABEL: section '.R_PPC64_TOC':
# DATABE-NEXT:    00000000 100281e8
