# REQUIRES: ppc

# RUN: llvm-mc -filetype=obj -triple=powerpc64le-unknown-linux %s -o %t.o
# RUN: ld.lld %t.o -o %t
# RUN: llvm-readelf -s %t | FileCheck --check-prefix=SYM %s
# RUN: llvm-readelf -S %t | FileCheck --check-prefix=SECTIONS %s
# RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck %s
# RUN: llvm-readobj -r %t | FileCheck --check-prefix=REL %s

# RUN: llvm-mc -filetype=obj -triple=powerpc64-unknown-linux %s -o %t.o
## IRELATIVE relocs relocating NOBITS .plt do not cause --check-dynamic-relocations errors.
# RUN: ld.lld %t.o -o %t --apply-dynamic-relocs --check-dynamic-relocations
# RUN: llvm-readelf -s %t | FileCheck --check-prefix=SYM %s
# RUN: llvm-readelf -S %t | FileCheck --check-prefix=SECTIONS %s
# RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck %s
# RUN: llvm-readobj -r %t | FileCheck --check-prefix=REL %s

# SYM: Value            Size Type   Bind   Vis     Ndx
# SYM: 0000000010028298    0 NOTYPE LOCAL  HIDDEN    4 .TOC.
# SYM: 0000000010010268    0 FUNC   GLOBAL DEFAULT   3 ifunc1
# SYM: 0000000010010210    0 IFUNC  GLOBAL DEFAULT   2 ifunc2
# SYM: 0000000010010288    0 FUNC   GLOBAL DEFAULT   3 ifunc3

# SECTIONS: .plt NOBITS 00000000100302a0 0002a0 000018 00 WA 0 0 8

# __plt_ifunc - . = 0x10010218 - 0x10010208 = 16
# __plt_ifunc2 - . = 0x1001022c - 0x10010210 = 28
# CHECK: <_start>:
# CHECK-NEXT:                 addis 2, 12, 2
# CHECK-NEXT:                 addi 2, 2, -32636
# CHECK-NEXT: 1001021c:       bl 0x10010240
# CHECK-NEXT:                 ld 2, 24(1)
# CHECK-NEXT: 10010224:       bl 0x10010254
# CHECK-NEXT:                 ld 2, 24(1)
# CHECK-NEXT:                 addis 3, 2, -2
# CHECK-NEXT:                 addi 3, 3, 32720
# CHECK-NEXT:                 addis 3, 2, -2
# CHECK-NEXT:                 addi 3, 3, 32752

# .plt[1] - .TOC. = 0x100302a0+8 - 0x10028298 = (1<<16) - 32752
# CHECK: <__plt_ifunc2>:
# CHECK-NEXT:     std 2, 24(1)
# CHECK-NEXT:     addis 12, 2, 1
# CHECK-NEXT:     ld 12, -32752(12)
# CHECK-NEXT:     mtctr 12
# CHECK-NEXT:     bctr

# .plt[2] - .TOC. = 0x100302a0+16 - 0x10028298 = (1<<16) - 32744
# CHECK: <__plt_ifunc3>:
# CHECK-NEXT:     std 2, 24(1)
# CHECK-NEXT:     addis 12, 2, 1
# CHECK-NEXT:     ld 12, -32744(12)
# CHECK-NEXT:     mtctr 12
# CHECK-NEXT:     bctr
# CHECK-EMPTY:

## .glink has 3 IPLT entries for ifunc1, ifunc2 and ifunc3.
## ifunc2 and ifunc3 have the same code sequence as their PLT call stubs.
# CHECK:      Disassembly of section .glink:
# CHECK-EMPTY:
# CHECK-NEXT: 0000000010010268 <ifunc1>:
# CHECK-NEXT:     addis 12, 2, 1
# CHECK-NEXT:     ld 12, -32760(12)
# CHECK-NEXT:     mtctr 12
# CHECK-NEXT:     bctr
# CHECK-NEXT:     addis 12, 2, 1
# CHECK-NEXT:     ld 12, -32752(12)
# CHECK-NEXT:     mtctr 12
# CHECK-NEXT:     bctr
# CHECK-EMPTY:
# CHECK-NEXT: 0000000010010288 <ifunc3>:
# CHECK-NEXT:     addis 12, 2, 1
# CHECK-NEXT:     ld 12, -32744(12)
# CHECK-NEXT:     mtctr 12
# CHECK-NEXT:     bctr

## Check that we emit 3 R_PPC64_IRELATIVE in .rela.dyn.
# REL:      .rela.dyn {
# REL-NEXT:   0x100302A0 R_PPC64_IRELATIVE - 0x10010210
# REL-NEXT:   0x100302A8 R_PPC64_IRELATIVE - 0x10010210
# REL-NEXT:   0x100302B0 R_PPC64_IRELATIVE - 0x10010210
# REL-NEXT: }

.type ifunc1,@gnu_indirect_function
.type ifunc2,@gnu_indirect_function
.type ifunc3,@gnu_indirect_function
.globl ifunc1, ifunc2, ifunc3
ifunc1:
ifunc2:
ifunc3:
  blr

.global _start
.type   _start,@function

_start:
.Lfunc_gep0:
  addis 2, 12, .TOC.-.Lfunc_gep0@ha
  addi 2, 2, .TOC.-.Lfunc_gep0@l
.Lfunc_lep0:
  .localentry     _start, .Lfunc_lep0-.Lfunc_gep0

  ## ifunc1 is taken address.
  ## ifunc2 is called.
  ## ifunc3 is both taken address and called.
  ## We need to create IPLT entries in .glink for ifunc1 and ifunc3, change
  ## their types from STT_GNU_IFUNC to STT_FUNC, and set their st_shndx/st_value
  ## to their .glink entries. Technically we don't need an entry for ifunc2 in
  ## .glink, but we currently do that.
  bl ifunc2
  nop
  bl ifunc3
  nop

  addis 3, 2, ifunc1@toc@ha
  addi  3, 3, ifunc1@toc@l
  addis 3, 2, ifunc3@toc@ha
  addi  3, 3, ifunc3@toc@l
