# REQUIRES: ppc

# RUN: llvm-mc -filetype=obj -triple=powerpc64le-unknown-linux %s -o %t.o
# RUN: ld.lld %t.o -o %t
# RUN: llvm-nm %t | FileCheck --check-prefix=NM %s
# RUN: llvm-readelf -S %t | FileCheck --check-prefix=SECTIONS %s
# RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck %s
# RUN: llvm-readobj -r %t | FileCheck --check-prefix=REL %s

# RUN: llvm-mc -filetype=obj -triple=powerpc64-unknown-linux %s -o %t.o
# RUN: ld.lld %t.o -o %t
# RUN: llvm-nm %t | FileCheck --check-prefix=NM %s
# RUN: llvm-readelf -S %t | FileCheck --check-prefix=SECTIONS %s
# RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck %s
# RUN: llvm-readobj -r %t | FileCheck --check-prefix=REL %s

# NM-DAG: 0000000010028248 d .TOC.
# NM-DAG: 00000000100101f8 i ifunc
# NM-DAG: 00000000100101fc i ifunc2

# SECTIONS: .plt NOBITS 0000000010030250 000250 000010 00 WA 0 0 8

# __plt_ifunc - . = 0x10010218 - 0x10010208 = 16
# __plt_ifunc2 - . = 0x1001022c - 0x10010210 = 28
# CHECK: _start:
# CHECK-NEXT:                 addis 2, 12, 2
# CHECK-NEXT:                 addi 2, 2, -32696
# CHECK-NEXT: 10010208:       bl .+16
# CHECK-NEXT:                 ld 2, 24(1)
# CHECK-NEXT: 10010210:       bl .+28
# CHECK-NEXT:                 ld 2, 24(1)

# .plt[0] - .TOC. = 0x10030250 - 0x10028248 = (1<<16) - 32760
# CHECK: __plt_ifunc:
# CHECK-NEXT:     std 2, 24(1)
# CHECK-NEXT:     addis 12, 2, 1
# CHECK-NEXT:     ld 12, -32760(12)
# CHECK-NEXT:     mtctr 12
# CHECK-NEXT:     bctr

# .plt[1] - .TOC. = 0x10030250+8 - 0x10028248 = (1<<16) - 32752
# CHECK: __plt_ifunc2:
# CHECK-NEXT:     std 2, 24(1)
# CHECK-NEXT:     addis 12, 2, 1
# CHECK-NEXT:     ld 12, -32752(12)
# CHECK-NEXT:     mtctr 12
# CHECK-NEXT:     bctr

## Check that we emit 2 R_PPC64_IRELATIVE in .rela.dyn.
## glibc powerpc64 does not eagerly resolve R_PPC64_IRELATIVE if they are in .rela.plt.
# REL:      .rela.dyn {
# REL-NEXT:   0x10030250 R_PPC64_IRELATIVE - 0x100101F8
# REL-NEXT:   0x10030258 R_PPC64_IRELATIVE - 0x100101FC
# REL-NEXT: }

.type ifunc STT_GNU_IFUNC
.globl ifunc
ifunc:
  nop

.type ifunc2 STT_GNU_IFUNC
.globl ifunc2
ifunc2:
  nop

.global _start
.type   _start,@function

_start:
.Lfunc_gep0:
  addis 2, 12, .TOC.-.Lfunc_gep0@ha
  addi 2, 2, .TOC.-.Lfunc_gep0@l
.Lfunc_lep0:
  .localentry     _start, .Lfunc_lep0-.Lfunc_gep0
  bl ifunc
  nop
  bl ifunc2
  nop
