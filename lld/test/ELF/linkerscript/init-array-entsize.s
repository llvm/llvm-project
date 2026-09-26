# REQUIRES: x86
# RUN: split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/t64.s -o %t64.o
# RUN: llvm-mc -filetype=obj -triple=i386 %t/t32.s -o %t32.o
# RUN: yaml2obj %t/entsize16.yaml -o %t16.o
# RUN: yaml2obj %t/conflict.yaml -o %tconflict.o

## init_array/fini_array/preinit_array sections hold tables of pointers, so
## sh_entsize should be the pointer size even when no input section provided
## one (e.g. sections declared in a linker script). This matches GNU ld.
# RUN: ld.lld -T %t/script.lds %t64.o -o %t64
# RUN: llvm-readelf -S %t64 | FileCheck %s --check-prefix=CHECK64

# RUN: ld.lld -m elf_i386 -T %t/script.lds %t32.o -o %t32
# RUN: llvm-readelf -S %t32 | FileCheck %s --check-prefix=CHECK32

## An explicit input entry size is kept as is.
# RUN: ld.lld -T %t/propagate.lds %t16.o -o %t16
# RUN: llvm-readelf -S %t16 | FileCheck %s --check-prefix=PROP

## Conflicting input entry sizes record no entry size (0). sh_size 24 is a
## multiple of 8, so fall back to 8, matching GNU ld.
# RUN: ld.lld -T %t/propagate.lds %tconflict.o -o %tconflict
# RUN: llvm-readelf -S %tconflict | FileCheck %s --check-prefix=CONFLICT

## A 0-byte section trivially has a multiple of the word size as sh_size, so
## set sh_entsize to the word size. (GNU ld removes empty script sections.)
# CHECK64:      .init_array INIT_ARRAY [[#%x,]] [[#%x,]] 000000 08
# CHECK64:      .fini_array FINI_ARRAY [[#%x,]] [[#%x,]] 000000 08
## 9-byte contents are not a table of pointers: keep sh_entsize 0.
# CHECK64:      .preinit_array PREINIT_ARRAY [[#%x,]] [[#%x,]] 000009 00

# CHECK32:      .init_array INIT_ARRAY [[#%x,]] [[#%x,]] 000000 04
# CHECK32:      .fini_array FINI_ARRAY [[#%x,]] [[#%x,]] 000000 04
# CHECK32:      .preinit_array PREINIT_ARRAY [[#%x,]] [[#%x,]] 000009 00

# PROP:         .init_array INIT_ARRAY [[#%x,]] [[#%x,]] 000010 10

# CONFLICT:     .init_array INIT_ARRAY [[#%x,]] [[#%x,]] 000018 08

#--- t64.s
.globl _start
_start:
  movq __init_array_start@GOTPCREL(%rip), %rax
  movq __fini_array_start@GOTPCREL(%rip), %rax
  ret

#--- t32.s
.globl _start
_start:
  movl $__init_array_start, %eax
  movl $__fini_array_start, %eax
  movl $__preinit_array_start, %eax
  ret

#--- script.lds
SECTIONS {
  .text : { *(.text) }
  .init_array (TYPE=SHT_INIT_ARRAY) : { *(.init_array) }
  .fini_array (TYPE=SHT_FINI_ARRAY) : { *(.fini_array) }
  .preinit_array (TYPE=SHT_PREINIT_ARRAY) : { QUAD(1) BYTE(0) }
}

#--- propagate.lds
SECTIONS {
  .init_array : { *(.init_array .init_array.*) }
}

#--- entsize16.yaml
--- !ELF
FileHeader:
  Class:   ELFCLASS64
  Data:    ELFDATA2LSB
  Type:    ET_REL
  Machine: EM_X86_64
Sections:
  - Name: .text
    Type: SHT_PROGBITS
    Flags: [ SHF_ALLOC, SHF_EXECINSTR ]
    Content: C3
  - Name: .init_array
    Type: SHT_INIT_ARRAY
    Flags: [ SHF_ALLOC, SHF_WRITE ]
    EntSize: 16
    Content: "00000000000000000000000000000000"

#--- conflict.yaml
--- !ELF
FileHeader:
  Class:   ELFCLASS64
  Data:    ELFDATA2LSB
  Type:    ET_REL
  Machine: EM_X86_64
Sections:
  - Name: .text
    Type: SHT_PROGBITS
    Flags: [ SHF_ALLOC, SHF_EXECINSTR ]
    Content: C3
  - Name: .init_array
    Type: SHT_INIT_ARRAY
    Flags: [ SHF_ALLOC, SHF_WRITE ]
    EntSize: 8
    Content: "0000000000000000"
  - Name: .init_array.2
    Type: SHT_INIT_ARRAY
    Flags: [ SHF_ALLOC, SHF_WRITE ]
    EntSize: 16
    Content: "00000000000000000000000000000000"
