# REQUIRES: loongarch
## R_LARCH_RELAX is paired with another relocation at the same offset, and the
## ABI does not require the marker to follow the relocation it annotates.  lld
## must not relax only part of a TLS LE sequence when the marker of the
## consumer precedes it: that deletes the instructions producing the address
## while the consumer keeps reading the incoming register.

# RUN: yaml2obj %s -o %t.o
# RUN: ld.lld %t.o -o %t
# RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn %t | FileCheck %s --check-prefixes=CANONICAL,HINT-FIRST

# CANONICAL-LABEL: <canonical>:
# CANONICAL-NEXT:    addi.d  $a0, $tp, 0
# CANONICAL-NEXT:    ret
# HINT-FIRST-LABEL: <hint_first>:
# HINT-FIRST-NEXT:    addi.d  $a0, $tp, 0
# HINT-FIRST-NEXT:    ret

--- !ELF
FileHeader:
  Class:   ELFCLASS64
  Data:    ELFDATA2LSB
  Type:    ET_REL
  Machine: EM_LOONGARCH
  Flags:   [ EF_LOONGARCH_ABI_DOUBLE_FLOAT, EF_LOONGARCH_OBJABI_V1 ]
Sections:
## lu12i.w $a0, 0 / add.d $a0, $a0, $tp / addi.d $a0, $a0, 0 / ret
  - Name: .text.canonical
    Type: SHT_PROGBITS
    Flags: [ SHF_ALLOC, SHF_EXECINSTR ]
    Content: "04000014848810008400C0022000004C"
  - Name: .text.hint_first
    Type: SHT_PROGBITS
    Flags: [ SHF_ALLOC, SHF_EXECINSTR ]
    Content: "04000014848810008400C0022000004C"
  - Name: .tbss
    Type: SHT_NOBITS
    Flags: [ SHF_ALLOC, SHF_WRITE, SHF_TLS ]
    AddressAlign: 8
    Size: 8
  - Name: .rela.text.canonical
    Type: SHT_RELA
    Link: .symtab
    Info: .text.canonical
    Relocations:
      - Offset: 0
        Symbol: tvar
        Type: R_LARCH_TLS_LE_HI20_R
      - Offset: 0
        Type: R_LARCH_RELAX
      - Offset: 4
        Symbol: tvar
        Type: R_LARCH_TLS_LE_ADD_R
      - Offset: 4
        Type: R_LARCH_RELAX
      - Offset: 8
        Symbol: tvar
        Type: R_LARCH_TLS_LE_LO12_R
      - Offset: 8
        Type: R_LARCH_RELAX
  - Name: .rela.text.hint_first
    Type: SHT_RELA
    Link: .symtab
    Info: .text.hint_first
    Relocations:
      - Offset: 0
        Symbol: tvar
        Type: R_LARCH_TLS_LE_HI20_R
      - Offset: 0
        Type: R_LARCH_RELAX
      - Offset: 4
        Symbol: tvar
        Type: R_LARCH_TLS_LE_ADD_R
      - Offset: 4
        Type: R_LARCH_RELAX
## The marker of the consumer precedes its relocation.
      - Offset: 8
        Type: R_LARCH_RELAX
      - Offset: 8
        Symbol: tvar
        Type: R_LARCH_TLS_LE_LO12_R
Symbols:
  - Name: canonical
    Section: .text.canonical
    Binding: STB_GLOBAL
    Type: STT_FUNC
  - Name: hint_first
    Section: .text.hint_first
    Binding: STB_GLOBAL
    Type: STT_FUNC
  - Name: tvar
    Section: .tbss
    Binding: STB_GLOBAL
    Type: STT_TLS
    Size: 8
