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

# RUN: llvm-readelf -S -s %t
# RUN: llvm-readelf -S -s %t | FileCheck %s
#        [Nr] Name              Type            Address          Off    Size   ES Flg Lk Inf Al
# CHECK: [65275] .note.GNU-stack PROGBITS       0000000000000000 000040 000000 00      0   0  1
# CHECK: [65276] .symtab        SYMTAB          0000000000000000 000040 17e7b8 18     65279 65275  8
# CHECK: [65277] .symtab_shndx  SYMTAB SECTION INDICES 0000000000000000 17e7f8 03fbf4 04     65276   0  4
# CHECK: [65278] .shstrtab      STRTAB          0000000000000000 1be3ec 0e7734 00      0   0  1
# CHECK: [65279] .strtab        STRTAB          0000000000000000 2a5b20 00000c 00      0   0  1

# 5) Check we are able to represent symbol foo with section (.bar)
# CHECK: GLOBAL DEFAULT  65274 foo

.macro gen_sections4 x
  .section a\x
  .section b\x
  .section c\x
  .section d\x
.endm

.macro gen_sections8 x
  gen_sections4 a\x
  gen_sections4 b\x
.endm

.macro gen_sections16 x
  gen_sections8 a\x
  gen_sections8 b\x
.endm

.macro gen_sections32 x
  gen_sections16 a\x
  gen_sections16 b\x
.endm

.macro gen_sections64 x
  gen_sections32 a\x
  gen_sections32 b\x
.endm

.macro gen_sections128 x
  gen_sections64 a\x
  gen_sections64 b\x
.endm

.macro gen_sections256 x
  gen_sections128 a\x
  gen_sections128 b\x
.endm

.macro gen_sections512 x
  gen_sections256 a\x
  gen_sections256 b\x
.endm

.macro gen_sections1024 x
  gen_sections512 a\x
  gen_sections512 b\x
.endm

.macro gen_sections2048 x
  gen_sections1024 a\x
  gen_sections1024 b\x
.endm

.macro gen_sections4096 x
  gen_sections2048 a\x
  gen_sections2048 b\x
.endm

.macro gen_sections8192 x
  gen_sections4096 a\x
  gen_sections4096 b\x
.endm

.macro gen_sections16384 x
  gen_sections8192 a\x
  gen_sections8192 b\x
.endm

## 65,272 sections, fewer than SHN_LORESERVE after unused Output Sections
## removed, but SHT_SYMTAB_SHNDX may still be written.
gen_sections16384 a
gen_sections16384 b
gen_sections16384 c
gen_sections8192 d
gen_sections4096 e
gen_sections2048 f
gen_sections1024 g
gen_sections512  h
gen_sections128  i
gen_sections64   j
gen_sections32   k
gen_sections16   l
gen_sections8    m

.section .bar
.global foo
foo:

.section .text, "ax"
.global _start
_start:
