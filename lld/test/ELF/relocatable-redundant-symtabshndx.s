# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o %t.o
# RUN: ld.lld -r %t.o -o %t

## In rare cases, the number of output sections exceeds SHN_LORESERVE prior
## to removal of synthetic sections, and other empty output sections. Ideally
## lld would be able to remove the .symtab_shndx, but until then check that
## the .symtab_shndx is finalized, with a legal sh_link.

## Check we are able to link against relocatable file produced.
# RUN: ld.lld %t -o %t.out

## Check we emit a valid ELF SHT_SYMTAB_SHNDX.
# RUN: llvm-readobj --file-headers %t | FileCheck %s --check-prefix=HDR
# HDR:      ElfHeader {
# HDR:        SectionHeaderCount: 0 (65280)
# HDR-NEXT:   StringTableSectionIndex: 65278

## Check that:
## 1) 65278 is the index of .shstrtab section.
## 2) .symtab_shndx is linked with .symtab.
## 3) .symtab_shndx entry size and alignment == 4.
## 4) .symtab_shndx has size equal to
##    (sizeof(.symtab) / entsize(.symtab)) * entsize(.symtab_shndx) = 0x4 * 0x17e7b8 / 0x18 == 0x03fbf4

# RUN: llvm-readelf -S -s %t | FileCheck %s
#        [Nr] Name              Type            Address          Off    Size   ES Flg Lk Inf Al
# CHECK: [65275] .note.GNU-stack PROGBITS       0000000000000000 000040 000000 00      0   0  1
# CHECK-NEXT: [65276] .symtab        SYMTAB          0000000000000000 000040 17e7b8 18     65279 65275  8
# CHECK-NEXT: [65277] .symtab_shndx  SYMTAB SECTION INDICES 0000000000000000 17e7f8 03fbf4 04     65276   0  4
# CHECK-NEXT: [65278] .shstrtab      STRTAB          0000000000000000 1be3ec 05ceae 00      0   0  1
# CHECK-NEXT: [65279] .strtab        STRTAB          0000000000000000 21b29a 00000c 00      0   0  1

# 5) Check we are able to represent symbol foo with section (.bar)
# CHECK: GLOBAL DEFAULT  65274 foo
## 65,272 sections, fewer than SHN_LORESERVE after unused Output Sections
## removed, but SHT_SYMTAB_SHNDX may still be written.
.rept 65272
.section \+
.endr

.section .bar
.global foo
foo:

.section .text, "ax"
.global _start
_start:
