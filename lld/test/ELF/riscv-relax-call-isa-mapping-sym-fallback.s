# REQUIRES: riscv
## Fallback when R_RISCV_RELAX lacks a usable ISA mapping symbol: the linker uses
## the file-level EF_RISCV_RVC flag.
##   * f_null: null symbol (old assemblers); backward compatible, no warning.
##   * f_bad:  unparseable "$x<...>" string; warns once, then falls back.

# RUN: sed 's/EF_RISCV_FLAGS/EF_RISCV_RVC/' %s | yaml2obj -o %t.rvc.o
# RUN: sed 's/EF_RISCV_FLAGS//' %s | yaml2obj -o %t.norvc.o

## EF_RISCV_RVC set -> c.j; cleared -> jal.
# RUN: ld.lld %t.rvc.o -Ttext=0x10000 -o %t.rvc 2>&1 | FileCheck %s --check-prefix=WARN
# RUN: llvm-objdump -d --no-show-raw-insn -M no-aliases %t.rvc | FileCheck %s --check-prefix=RVC
# RUN: ld.lld %t.norvc.o -Ttext=0x10000 -o %t.norvc 2>&1 | FileCheck %s --check-prefix=WARN
# RUN: llvm-objdump -d --no-show-raw-insn -M no-aliases %t.norvc | FileCheck %s --check-prefix=NORVC

# WARN: warning: {{.*}}: R_RISCV_RELAX ISA mapping symbol '$xnotaparseablearchstring' has an unparseable ISA string:{{.*}}falling back to EF_RISCV_RVC

# RVC-LABEL: <f_null>:
# RVC-NEXT: {{.*}}: c.j {{.*}} <target>
# RVC-LABEL: <f_bad>:
# RVC-NEXT: {{.*}}: c.j {{.*}} <target>

# NORVC-LABEL: <f_null>:
# NORVC-NEXT: {{.*}}: jal zero, {{.*}} <target>
# NORVC-LABEL: <f_bad>:
# NORVC-NEXT: {{.*}}: jal zero, {{.*}} <target>

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
    AddressAlign: 0x2
    ## f_null: tail target ; f_bad: tail target ; target: c.ret
    Content:      '170300006700030017030000670003008280'
  - Name:         .rela.text
    Type:         SHT_RELA
    Flags:        [ SHF_INFO_LINK ]
    Link:         .symtab
    Info:         .text
    AddressAlign: 0x8
    Relocations:
      - Offset: 0x0
        Symbol: target
        Type:   R_RISCV_CALL_PLT
      - Offset: 0x0
        Type:   R_RISCV_RELAX
      - Offset: 0x8
        Symbol: target
        Type:   R_RISCV_CALL_PLT
      - Offset: 0x8
        Symbol: '$xnotaparseablearchstring'
        Type:   R_RISCV_RELAX
Symbols:
  - Name:    '$xnotaparseablearchstring'
    Section: .text
    Binding: STB_LOCAL
    Value:   0x0
  - Name:    f_null
    Section: .text
    Binding: STB_GLOBAL
    Value:   0x0
  - Name:    f_bad
    Section: .text
    Binding: STB_GLOBAL
    Value:   0x8
  - Name:    target
    Section: .text
    Binding: STB_GLOBAL
    Value:   0x10
