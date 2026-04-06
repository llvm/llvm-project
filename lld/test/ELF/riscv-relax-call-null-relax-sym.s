# REQUIRES: riscv
## Backward-compatibility test for R_RISCV_RELAX with a null symbol (symbol
## index 0, shown as '-' by llvm-readobj).  Older assemblers always emit
## R_RISCV_RELAX with a null symbol, so the linker must fall back to the
## file-level EF_RISCV_RVC flag to decide whether compressed-jump relaxation
## is available.

# RUN: sed 's/EF_RISCV_FLAGS/EF_RISCV_RVC/' %s | yaml2obj -o %t.rvc.o
# RUN: sed 's/EF_RISCV_FLAGS//' %s | yaml2obj -o %t.norvc.o

# RUN: ld.lld %t.rvc.o -Ttext=0x10000 -o %t.rvc
# RUN: llvm-objdump -d --no-show-raw-insn -M no-aliases %t.rvc | FileCheck %s --check-prefix=RVC

# RUN: ld.lld %t.norvc.o -Ttext=0x10000 -o %t.norvc
# RUN: llvm-objdump -d --no-show-raw-insn -M no-aliases %t.norvc | FileCheck %s --check-prefix=NORVC

## EF_RISCV_RVC set: fallback enables c.j relaxation.
# RVC-LABEL: <compat>:
# RVC-NEXT: {{.*}}: c.j {{.*}} <compat_target>

## EF_RISCV_RVC cleared: fallback picks rvc=false, so the relaxation lands on
## uncompressed jal.
# NORVC-LABEL: <compat>:
# NORVC-NEXT: {{.*}}: jal zero, {{.*}} <compat_target>

--- !ELF
FileHeader:
  Class:   ELFCLASS64
  Data:    ELFDATA2LSB
  Type:    ET_REL
  Machine: EM_RISCV
  Flags:   [ EF_RISCV_FLAGS ]
Sections:
  - Name:         .text
    Type:         SHT_PROGBITS
    Flags:        [ SHF_ALLOC, SHF_EXECINSTR ]
    AddressAlign: 0x4
    Content:      '170300006700030067800000'
  - Name:         .rela.text
    Type:         SHT_RELA
    Flags:        [ SHF_INFO_LINK ]
    Link:         .symtab
    Info:         .text
    AddressAlign: 0x8
    Relocations:
      - Offset: 0x0
        Symbol: compat_target
        Type:   R_RISCV_CALL_PLT
      - Offset: 0x0
        Type:   R_RISCV_RELAX
Symbols:
  - Name:    compat
    Section: .text
    Binding: STB_GLOBAL
    Value:   0x0
  - Name:    compat_target
    Section: .text
    Binding: STB_GLOBAL
    Value:   0x8
