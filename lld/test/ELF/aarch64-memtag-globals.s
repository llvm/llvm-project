# REQUIRES: aarch64

# RUN: rm -rf %t

## Ensure MTE globals doesn't work with REL (only RELA).
# RUN: yaml2obj %s -o %t.rel.o
# RUN: not ld.lld -shared --android-memtag-mode=sync %t.rel.o -o %t1.so 2>&1 | FileCheck %s --check-prefix=CHECK-RELA
# CHECK-RELA: non-RELA relocations are not allowed with memtag globals
--- !ELF
FileHeader:
  Class:           ELFCLASS64
  Data:            ELFDATA2LSB
  Type:            ET_REL
  Machine:         EM_AARCH64
  SectionHeaderStringTable: .strtab
Sections:
  - Name:            .text
    Type:            SHT_PROGBITS
    Flags:           [ SHF_ALLOC, SHF_EXECINSTR ]
    AddressAlign:    0x4
    Content:         '00'
  - Name:            .data
    Type:            SHT_PROGBITS
    Flags:           [ SHF_WRITE, SHF_ALLOC ]
    AddressAlign:    0x10
    Content:         '00'
  - Name:            .memtag.globals.static
    Type:            SHT_AARCH64_MEMTAG_GLOBALS_STATIC
    AddressAlign:    0x1
  - Name:            .rel.memtag.globals.static
    Type:            SHT_REL
    Flags:           [ SHF_INFO_LINK ]
    Link:            .symtab
    AddressAlign:    0x8
    Info:            .memtag.globals.static
    Relocations:
      - Symbol:          four
        Type:            R_AARCH64_NONE
  - Type:            SectionHeaderTable
    Sections:
      - Name:            .strtab
      - Name:            .text
      - Name:            .data
      - Name:            .memtag.globals.static
      - Name:            .rel.memtag.globals.static
      - Name:            .symtab
Symbols:
  - Name:            four
    Type:            STT_OBJECT
    Section:         .data
    Binding:         STB_GLOBAL
    Value:           0x0
    Size:            0x10

## Functional testing for MTE globals.
# RUN: split-file %S/Inputs/aarch64-memtag-globals.s %t
# RUN: llvm-mc --filetype=obj -triple=aarch64-none-linux-android \
# RUN:   %t/input_1.s -o %t1.o
# RUN: llvm-mc --filetype=obj -triple=aarch64-none-linux-android \
# RUN:   %t/input_2.s -o %t2.o
# RUN: ld.lld -shared --android-memtag-mode=sync %t1.o %t2.o -o %t.so

## Normally relocations are printed before the symbol tables, so reorder it a
## bit to make it easier on matching addresses of relocations up with the
## symbols.
# RUN: llvm-readelf %t.so -s > %t.out
# RUN: llvm-readelf %t.so --section-headers --relocs --memtag >> %t.out
# RUN: FileCheck %s < %t.out
# RUN: llvm-objdump -Dz %t.so | FileCheck %s --check-prefix=CHECK-SPECIAL-RELOCS

## And ensure that --apply-dynamic-relocs is banned.
# RUN: not ld.lld --apply-dynamic-relocs -shared --android-memtag-mode=sync \
# RUN:   %t1.o %t2.o -o %t1.so 2>&1 | FileCheck %s --check-prefix=CHECK-DYNRELOC
# CHECK-DYNRELOC: --apply-dynamic-relocs cannot be used with MTE globals

## And ensure that fully-static executables are banned.
# RUN: not ld.lld --static --android-memtag-mode=sync \
# RUN:   %t1.o %t2.o -o %t1.so 2>&1 | FileCheck %s --check-prefix=CHECK-NOSTATIC
# CHECK-NOSTATIC: --android-memtag-mode is incompatible with fully-static executables (-static)

# CHECK:     Symbol table '.dynsym' contains
# CHECK-DAG: [[#%x,GLOBAL:]] 32 OBJECT GLOBAL DEFAULT [[#]] global{{$}}
# CHECK-DAG: [[#%x,GLOBAL_UNTAGGED:]] 4 OBJECT GLOBAL DEFAULT [[#]] global_untagged{{$}}
# CHECK-DAG: [[#%x,CONST_GLOBAL:]] 4 OBJECT GLOBAL DEFAULT [[#]] const_global{{$}}
# CHECK-DAG: [[#%x,GLOBAL_EXTERN:]] 16 OBJECT GLOBAL DEFAULT [[#]] global_extern{{$}}
# CHECK-DAG: [[#%x,GLOBAL_EXTERN_UNTAGGED:]] 4 OBJECT GLOBAL DEFAULT [[#]] global_extern_untagged{{$}}
# CHECK-DAG: 0 NOTYPE GLOBAL DEFAULT UND global_extern_outside_this_dso{{$}}
# CHECK-DAG: [[#%x,GLOBAL_EXTERN_CONST_DEFINITION_BUT_NONCONST_IMPORT:]] 4 OBJECT GLOBAL DEFAULT [[#]] global_extern_untagged_definition_but_tagged_import{{$}}
# CHECK-DAG: [[#%x,GLOBAL_EXTERN_UNTAGGED_DEFINITION_BUT_TAGGED_IMPORT:]] 4 OBJECT GLOBAL DEFAULT [[#]] global_extern_const_definition_but_nonconst_import{{$}}
# CHECK-DAG: [[#%x,POINTER_TO_GLOBAL:]] 16 OBJECT GLOBAL DEFAULT [[#]] pointer_to_global{{$}}
# CHECK-DAG: [[#%x,POINTER_INSIDE_GLOBAL:]] 16 OBJECT GLOBAL DEFAULT [[#]] pointer_inside_global{{$}}
# CHECK-DAG: [[#%x,POINTER_TO_GLOBAL_END:]] 16 OBJECT GLOBAL DEFAULT [[#]] pointer_to_global_end{{$}}
# CHECK-DAG: [[#%x,POINTER_PAST_GLOBAL_END:]] 16 OBJECT GLOBAL DEFAULT [[#]] pointer_past_global_end{{$}}
# CHECK-DAG: [[#%x,POINTER_TO_GLOBAL_UNTAGGED:]] 16 OBJECT GLOBAL DEFAULT [[#]] pointer_to_global_untagged{{$}}
# CHECK-DAG: [[#%x,POINTER_TO_CONST_GLOBAL:]] 16 OBJECT GLOBAL DEFAULT [[#]] pointer_to_const_global{{$}}
# CHECK-DAG: [[#%x,POINTER_TO_HIDDEN_CONST_GLOBAL:]] 16 OBJECT GLOBAL DEFAULT [[#]] pointer_to_hidden_const_global{{$}}
# CHECK-DAG: [[#%x,POINTER_TO_HIDDEN_GLOBAL:]] 16 OBJECT GLOBAL DEFAULT [[#]] pointer_to_hidden_global{{$}}
# CHECK-DAG: [[#%x,POINTER_TO_HIDDEN_GLOBAL_END:]] 16 OBJECT GLOBAL DEFAULT [[#]] pointer_to_hidden_global_end{{$}}
# CHECK-DAG: [[#%x,POINTER_PAST_HIDDEN_GLOBAL_END:]] 16 OBJECT GLOBAL DEFAULT [[#]] pointer_past_hidden_global_end{{$}}
# CHECK-DAG: [[#%x,POINTER_TO_HIDDEN_ATTR_GLOBAL:]] 16 OBJECT GLOBAL DEFAULT [[#]] pointer_to_hidden_attr_global{{$}}
# CHECK-DAG: [[#%x,POINTER_TO_HIDDEN_ATTR_CONST_GLOBAL:]] 16 OBJECT GLOBAL DEFAULT [[#]] pointer_to_hidden_attr_const_global{{$}}
# CHECK-DAG: [[#%x,POINTER_TO_GLOBAL_EXTERN:]] 16 OBJECT GLOBAL DEFAULT [[#]] pointer_to_global_extern{{$}}
# CHECK-DAG: [[#%x,POINTER_TO_GLOBAL_EXTERN_UNTAGGED:]] 16 OBJECT GLOBAL DEFAULT [[#]] pointer_to_global_extern_untagged{{$}}
# CHECK-DAG: [[#%x,POINTER_TO_GLOBAL_EXTERN_OUTSIDE_THIS_DSO:]] 16 OBJECT GLOBAL DEFAULT [[#]] pointer_to_global_extern_outside_this_dso{{$}}
# CHECK-DAG: [[#%x,POINTER_TO_GLOBAL_EXTERN_CONST_DEFINITION_BUT_NONCONST_IMPORT:]] 16 OBJECT GLOBAL DEFAULT [[#]] pointer_to_global_extern_const_definition_but_nonconst_import{{$}}
# CHECK-DAG: [[#%x,POINTER_TO_GLOBAL_EXTERN_UNTAGGED_DEFINITION_BUT_TAGGED_IMPORT:]] 16 OBJECT GLOBAL DEFAULT [[#]] pointer_to_global_extern_untagged_definition_but_tagged_import{{$}}

# CHECK:     Symbol table '.symtab' contains
# CHECK-DAG: [[#%x,HIDDEN_CONST_GLOBAL:]] 4 OBJECT LOCAL DEFAULT [[#]] hidden_const_global{{$}}
# CHECK-DAG: [[#%x,HIDDEN_GLOBAL:]] 16 OBJECT LOCAL DEFAULT [[#]] hidden_global{{$}}
# CHECK-DAG: [[#%x,HIDDEN_ATTR_GLOBAL:]] 16 OBJECT LOCAL HIDDEN [[#]] hidden_attr_global{{$}}
# CHECK-DAG: [[#%x,HIDDEN_ATTR_CONST_GLOBAL:]] 4 OBJECT LOCAL HIDDEN [[#]] hidden_attr_const_global{{$}}

# CHECK:     Section Headers:
# CHECK:     .memtag.globals.dynamic AARCH64_MEMTAG_GLOBALS_DYNAMIC
# CHECK-NOT: .memtag.globals.static
# CHECK-NOT: AARCH64_MEMTAG_GLOBALS_STATIC

# CHECK: Relocation section '.rela.dyn'
# CHECK-DAG: [[#POINTER_TO_GLOBAL]] {{.*}} R_AARCH64_ABS64 {{.*}} global + 0
# CHECK-DAG: [[#POINTER_INSIDE_GLOBAL]] {{.*}} R_AARCH64_ABS64 {{.*}} global + 11
# CHECK-DAG: [[#POINTER_TO_GLOBAL_END]] {{.*}} R_AARCH64_ABS64 {{.*}} global + 1e
# CHECK-DAG: [[#POINTER_PAST_GLOBAL_END]] {{.*}} R_AARCH64_ABS64 {{.*}} global + 30
# CHECK-DAG: [[#POINTER_TO_GLOBAL_UNTAGGED]] {{.*}} R_AARCH64_ABS64 {{.*}} global_untagged + 0
# CHECK-DAG: [[#POINTER_TO_CONST_GLOBAL]] {{.*}} R_AARCH64_ABS64 {{.*}} const_global + 0

## RELATIVE relocations.
# CHECK-DAG: [[#POINTER_TO_HIDDEN_CONST_GLOBAL]] {{.*}} R_AARCH64_RELATIVE {{0*}}[[#HIDDEN_CONST_GLOBAL]]
# CHECK-DAG: [[#POINTER_TO_HIDDEN_GLOBAL]] {{.*}} R_AARCH64_RELATIVE {{0*}}[[#HIDDEN_GLOBAL]]

## AArch64 MemtagABI special RELATIVE relocation semantics, where the offset is encoded in the place.
# CHECK-DAG:                 [[#POINTER_TO_HIDDEN_GLOBAL_END]] {{.*}} R_AARCH64_RELATIVE {{0*}}[[#HIDDEN_GLOBAL + 12]]
# CHECK-SPECIAL-RELOCS:      <pointer_to_hidden_global_end>:
# CHECK-SPECIAL-RELOCS-NEXT:   .word 0x00000000
# CHECK-SPECIAL-RELOCS-NEXT:   .word 0x00000000
# CHECK-DAG:                 [[#POINTER_PAST_HIDDEN_GLOBAL_END]] {{.*}} R_AARCH64_RELATIVE {{0*}}[[#HIDDEN_GLOBAL + 16]]
# CHECK-SPECIAL-RELOCS:      <pointer_past_hidden_global_end>:
# CHECK-SPECIAL-RELOCS-NEXT:   .word 0xfffffff0
# CHECK-SPECIAL-RELOCS-NEXT:   .word 0xffffffff

## More ABS64 relocations.
# CHECK-DAG: [[#POINTER_TO_GLOBAL_EXTERN]] {{[0-9a-f]+}} R_AARCH64_ABS64 {{[0-9a-f]+}} global_extern + 0
# CHECK-DAG: [[#POINTER_TO_GLOBAL_EXTERN_UNTAGGED]] {{[0-9a-f]+}} R_AARCH64_ABS64 {{[0-9a-f]+}} global_extern_untagged + 0
# CHECK-DAG: [[#POINTER_TO_GLOBAL_EXTERN_OUTSIDE_THIS_DSO]] {{[0-9a-f]+}} R_AARCH64_ABS64 {{[0-9a-f]+}} global_extern_outside_this_dso + 0
# CHECK-DAG: [[#POINTER_TO_GLOBAL_EXTERN_CONST_DEFINITION_BUT_NONCONST_IMPORT]] {{[0-9a-f]+}} R_AARCH64_ABS64 {{[0-9a-f]+}} global_extern_const_definition_but_nonconst_import + 0
# CHECK-DAG: [[#POINTER_TO_GLOBAL_EXTERN_UNTAGGED_DEFINITION_BUT_TAGGED_IMPORT]] {{[0-9a-f]+}} R_AARCH64_ABS64 {{[0-9a-f]+}} global_extern_untagged_definition_but_tagged_import + 0

# CHECK:      Memtag Dynamic Entries
# CHECK-NEXT: AARCH64_MEMTAG_MODE: Synchronous (0)
# CHECK-NEXT: AARCH64_MEMTAG_HEAP: Disabled (0)
# CHECK-NEXT: AARCH64_MEMTAG_STACK: Disabled (0)
# CHECK-NEXT: AARCH64_MEMTAG_GLOBALS: 0x{{[0-9a-f]+}}
# CHECK-NEXT: AARCH64_MEMTAG_GLOBALSSZ: 23

# CHECK:      Memtag Android Note
# CHECK-NEXT: Tagging Mode: SYNC
# CHECK-NEXT: Heap: Disabled
# CHECK-NEXT: Stack: Disabled

## Global variable order hopefully isn't too brittle of a test here, but this allows us to make sure
## that we have all the global variables we expect, and no more.
# CHECK:      Memtag Global Descriptors:
# CHECK-NEXT: 0x[[#POINTER_TO_GLOBAL]]: 0x10
# CHECK-NEXT: 0x[[#POINTER_INSIDE_GLOBAL]]: 0x10
# CHECK-NEXT: 0x[[#POINTER_TO_GLOBAL_END]]: 0x10
# CHECK-NEXT: 0x[[#POINTER_PAST_GLOBAL_END]]: 0x10
# CHECK-NEXT: 0x[[#POINTER_TO_GLOBAL_UNTAGGED]]: 0x10
# CHECK-NEXT: 0x[[#POINTER_TO_CONST_GLOBAL]]: 0x10
# CHECK-NEXT: 0x[[#POINTER_TO_HIDDEN_CONST_GLOBAL]]: 0x10
# CHECK-NEXT: 0x[[#POINTER_TO_HIDDEN_GLOBAL]]: 0x10
# CHECK-NEXT: 0x[[#POINTER_TO_HIDDEN_ATTR_GLOBAL]]: 0x10
# CHECK-NEXT: 0x[[#POINTER_TO_HIDDEN_ATTR_CONST_GLOBAL]]: 0x10
# CHECK-NEXT: 0x[[#POINTER_TO_HIDDEN_GLOBAL_END]]: 0x10
# CHECK-NEXT: 0x[[#POINTER_PAST_HIDDEN_GLOBAL_END]]: 0x10
# CHECK-NEXT: 0x[[#POINTER_TO_GLOBAL_EXTERN]]: 0x10
# CHECK-NEXT: 0x[[#POINTER_TO_GLOBAL_EXTERN_UNTAGGED]]: 0x10
# CHECK-NEXT: 0x[[#POINTER_TO_GLOBAL_EXTERN_OUTSIDE_THIS_DSO]]: 0x10
# CHECK-NEXT: 0x[[#POINTER_TO_GLOBAL_EXTERN_CONST_DEFINITION_BUT_NONCONST_IMPORT]]: 0x10
# CHECK-NEXT: 0x[[#POINTER_TO_GLOBAL_EXTERN_UNTAGGED_DEFINITION_BUT_TAGGED_IMPORT]]: 0x10
# CHECK-NEXT: 0x[[#GLOBAL]]: 0x20
# CHECK-NEXT: 0x[[#HIDDEN_ATTR_GLOBAL]]: 0x10
# CHECK-NEXT: 0x[[#HIDDEN_GLOBAL]]: 0x10
# CHECK-NEXT: 0x[[#GLOBAL_EXTERN]]: 0x10
# CHECK-NOT:  0x
