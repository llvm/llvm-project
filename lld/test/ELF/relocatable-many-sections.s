# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o %t.o
# RUN: ld.lld -r %t.o -o %t
# RUN: llvm-readobj -file-headers %t | FileCheck %s

## Check we are able to emit a valid ELF header when
## sections amount is greater than SHN_LORESERVE.
# CHECK:      ElfHeader {
# CHECK:        SectionHeaderCount: 0 (65541)
# CHECK-NEXT:   StringTableSectionIndex: 65535 (65539)

## Check that 65539 is really the index of .shstrtab section.
# RUN: llvm-objdump -section-headers -section=.shstrtab %t \
# RUN:   | FileCheck %s --check-prefix=SHSTRTAB
# SHSTRTAB:      Sections:
# SHSTRTAB-NEXT:  Idx   Name
# SHSTRTAB-NEXT:  65539 .shstrtab

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

gen_sections16384 a
gen_sections16384 b
gen_sections16384 c
gen_sections16384 d

.global _start
_start:
