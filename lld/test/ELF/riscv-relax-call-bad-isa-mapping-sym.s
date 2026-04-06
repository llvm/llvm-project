# REQUIRES: riscv
## Verify that when R_RISCV_RELAX references an ISA mapping symbol whose
## "$x<...>" suffix cannot be parsed as a normalized ISA string, the linker
## emits a warning and falls back to the file-level EF_RISCV_RVC flag.

# RUN: yaml2obj %s -o %t.o
# RUN: ld.lld %t.o -Ttext=0x10000 -o %t 2>&1 | FileCheck %s --check-prefix=WARN
# RUN: llvm-objdump -d --no-show-raw-insn -M no-aliases %t | FileCheck %s

# WARN: warning: {{.*}}: R_RISCV_RELAX ISA mapping symbol '$xnotaparseablearchstring' has an unparseable ISA string:{{.*}}falling back to EF_RISCV_RVC

## EF_RISCV_RVC is set on the object, so the fallback relaxes to c.j.
# CHECK-LABEL: <compat>:
# CHECK-NEXT: {{.*}}: c.j {{.*}} <compat_target>

--- !ELF
FileHeader:
  Class:   ELFCLASS64
  Data:    ELFDATA2LSB
  Type:    ET_REL
  Machine: EM_RISCV
  Flags:   [ EF_RISCV_RVC ]
Sections:
  - Name:         .text
    Type:         SHT_PROGBITS
    Flags:        [ SHF_ALLOC, SHF_EXECINSTR ]
    AddressAlign: 0x2
    ## tail compat_target ; c.ret
    Content:      '17030000670003008280'
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
        Symbol: '$xnotaparseablearchstring'
        Type:   R_RISCV_RELAX
Symbols:
  - Name:    '$xnotaparseablearchstring'
    Section: .text
    Binding: STB_LOCAL
    Value:   0x0
  - Name:    compat
    Section: .text
    Binding: STB_GLOBAL
    Value:   0x0
  - Name:    compat_target
    Section: .text
    Binding: STB_GLOBAL
    Value:   0x8
