# REQUIRES: x86
## A SHF_LINK_ORDER section whose linked-to section is discarded is discarded
## with it. Without that, the section is emitted with an sh_link that no longer
## designates an output section and the link fails.

# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 a.s -o a.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 b.s -o b.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 main.s -o main.o

## b.o's group is discarded, and so is the __patchable_function_entries section
## that points into it. Only a.o's entry is left, with an sh_link
## to the surviving .text.
# RUN: ld.lld main.o a.o b.o -o out
# RUN: llvm-readelf -S out | FileCheck %s \
# RUN:     --implicit-check-not='{{ }}__patchable_function_entries '
# CHECK: [[#%u,TEXT:]]] .text PROGBITS
# CHECK: {{ }}__patchable_function_entries PROGBITS {{.*}} 000008 00 WAL [[#%u,TEXT]]

## Same in a relocatable link and with --emit-relocs, where the relocation
## sections are copied to the output and must not be left with a dangling
## sh_info. Exactly one of each survives, and the surviving relocation section
## points at the surviving __patchable_function_entries.
# RUN: ld.lld -r main.o a.o b.o -o out.ro
# RUN: llvm-readelf -S out.ro | FileCheck %s --check-prefix=REL \
# RUN:     --implicit-check-not='{{ }}__patchable_function_entries ' \
# RUN:     --implicit-check-not='{{ }}.rela__patchable_function_entries '
# REL: [[#%u,PFE:]]] __patchable_function_entries PROGBITS
# REL: {{ }}.rela__patchable_function_entries RELA {{.*}} I {{[0-9]+}} [[#%u,PFE]]

# RUN: ld.lld --emit-relocs main.o a.o b.o -o out.er
# RUN: llvm-readelf -S out.er | FileCheck %s --check-prefix=REL \
# RUN:     --implicit-check-not='{{ }}__patchable_function_entries ' \
# RUN:     --implicit-check-not='{{ }}.rela__patchable_function_entries '

## A relocation section that precedes the SHF_LINK_ORDER section it relocates
## must be dropped along with it. yaml2obj is used because assemblers place
## relocation sections last.
# RUN: yaml2obj rev.yaml -o rev.o
# RUN: ld.lld -r main.o a.o rev.o -o out.rev
# RUN: llvm-readelf -S out.rev | FileCheck %s --check-prefix=REL \
# RUN:     --implicit-check-not='{{ }}__patchable_function_entries ' \
# RUN:     --implicit-check-not='{{ }}.rela__patchable_function_entries '

## A relocation section has no input section yet at this point, and the loop
## uses sh_link, while a relocation section names its target with sh_info.
## Discarding one on sh_link would silently drop its relocations.
## .rela.data.live has SHF_LINK_ORDER and an sh_link to the discarded group,
## but relocates the live .data.live, which must still get its relocation.
# RUN: yaml2obj reloc.yaml -o reloc.o
# RUN: ld.lld main.o a.o reloc.o -o out.reloc
# RUN: llvm-readelf -x .data out.reloc | FileCheck %s --check-prefix=RELOC
# RELOC:     Hex dump of section '.data':
# RELOC-NOT: 00000000 00000000

## A linked-to section may itself be SHF_LINK_ORDER. lo_a links to lo_b, which
## links to lo_c, which links to a discarded section, so all three go. Handling
## only one level fails the link with "sh_link points to discarded section
## <internal>:()". GNU ld follows the chain as well.
##
## The same object has a cycle and a self-link, which no rewrite of the loop may
## hang on. No member of a cycle reaches a discarded section, so all three stay.
# RUN: yaml2obj chain.yaml -o chain.o
# RUN: ld.lld main.o a.o chain.o -o out.chain
# RUN: llvm-readelf -S out.chain | FileCheck %s --check-prefix=CHAIN \
# RUN:     --implicit-check-not='{{ }}lo_a ' \
# RUN:     --implicit-check-not='{{ }}lo_b ' \
# RUN:     --implicit-check-not='{{ }}lo_c '
# CHAIN-DAG: {{ }}lo_self PROGBITS
# CHAIN-DAG: {{ }}lo_x PROGBITS
# CHAIN-DAG: {{ }}lo_y PROGBITS

#--- a.s
.globl fa
fa:
  retq

.section .text.P,"axG",@progbits,P,comdat
.globl P
P:
  retq

.section __patchable_function_entries,"awo",@progbits,.text.P
  .quad P

#--- b.s
.globl fb
fb:
  retq

.section .text.P,"axG",@progbits,P,comdat
.globl P
P:
  retq

.section __patchable_function_entries,"awo",@progbits,.text.P
  .quad P

#--- main.s
.globl _start
_start:
  callq P
  callq fa
  retq

#--- rev.yaml
--- !ELF
FileHeader:
  Class:   ELFCLASS64
  Data:    ELFDATA2LSB
  Type:    ET_REL
  Machine: EM_X86_64
Sections:
  - Name:    .text.P
    Type:    SHT_PROGBITS
    Flags:   [ SHF_ALLOC, SHF_EXECINSTR, SHF_GROUP ]
    Size:    1
  - Name:    .group
    Type:    SHT_GROUP
    Link:    .symtab
    Info:    P
    Members:
      - SectionOrType: GRP_COMDAT
      - SectionOrType: .text.P
  - Name:    .rela__patchable_function_entries
    Type:    SHT_RELA
    Link:    .symtab
    Info:    __patchable_function_entries
    Relocations:
      - Offset: 0
        Symbol: P
        Type:   R_X86_64_64
  - Name:    __patchable_function_entries
    Type:    SHT_PROGBITS
    Flags:   [ SHF_ALLOC, SHF_WRITE, SHF_LINK_ORDER ]
    Link:    .text.P
    Size:    8
Symbols:
  - Name:    P
    Section: .text.P
    Binding: STB_GLOBAL

#--- reloc.yaml
--- !ELF
FileHeader:
  Class:   ELFCLASS64
  Data:    ELFDATA2LSB
  Type:    ET_REL
  Machine: EM_X86_64
Sections:
  - Name:    .text.P
    Type:    SHT_PROGBITS
    Flags:   [ SHF_ALLOC, SHF_EXECINSTR, SHF_GROUP ]
    Size:    1
  - Name:    .group
    Type:    SHT_GROUP
    Link:    .symtab
    Info:    P
    Members:
      - SectionOrType: GRP_COMDAT
      - SectionOrType: .text.P
  - Name:    .data.live
    Type:    SHT_PROGBITS
    Flags:   [ SHF_ALLOC, SHF_WRITE ]
    Size:    8
  - Name:    .rela.data.live
    Type:    SHT_RELA
    Flags:   [ SHF_LINK_ORDER ]
    Link:    .text.P
    Info:    .data.live
    Relocations:
      - Offset: 0
        Symbol: fa
        Type:   R_X86_64_64
Symbols:
  - Name:    P
    Section: .text.P
    Binding: STB_GLOBAL
  - Name:    fa
    Binding: STB_GLOBAL

#--- chain.yaml
--- !ELF
FileHeader:
  Class:   ELFCLASS64
  Data:    ELFDATA2LSB
  Type:    ET_REL
  Machine: EM_X86_64
Sections:
  - Name:    .text.P
    Type:    SHT_PROGBITS
    Flags:   [ SHF_ALLOC, SHF_EXECINSTR, SHF_GROUP ]
    Size:    1
  - Name:    .group
    Type:    SHT_GROUP
    Link:    .symtab
    Info:    P
    Members:
      - SectionOrType: GRP_COMDAT
      - SectionOrType: .text.P
  - Name:    lo_a
    Type:    SHT_PROGBITS
    Flags:   [ SHF_ALLOC, SHF_LINK_ORDER ]
    Link:    lo_b
    Size:    8
  - Name:    lo_b
    Type:    SHT_PROGBITS
    Flags:   [ SHF_ALLOC, SHF_LINK_ORDER ]
    Link:    lo_c
    Size:    8
  - Name:    lo_c
    Type:    SHT_PROGBITS
    Flags:   [ SHF_ALLOC, SHF_LINK_ORDER ]
    Link:    .text.P
    Size:    8
  - Name:    lo_self
    Type:    SHT_PROGBITS
    Flags:   [ SHF_ALLOC, SHF_LINK_ORDER ]
    Link:    lo_self
    Size:    8
  - Name:    lo_x
    Type:    SHT_PROGBITS
    Flags:   [ SHF_ALLOC, SHF_LINK_ORDER ]
    Link:    lo_y
    Size:    8
  - Name:    lo_y
    Type:    SHT_PROGBITS
    Flags:   [ SHF_ALLOC, SHF_LINK_ORDER ]
    Link:    lo_x
    Size:    8
Symbols:
  - Name:    P
    Section: .text.P
    Binding: STB_GLOBAL
