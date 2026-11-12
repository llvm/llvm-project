# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 %p/Inputs/shared.s -o %t2.o
# RUN: ld.lld -shared -soname=t2 %t2.o -o %t2.so

# RUN: ld.lld %t.o %t2.so -o %t
# RUN: ld.lld -shared %t.o %t2.so -o %t.so
# RUN: llvm-readelf -S -r %t | FileCheck %s --check-prefix=CHECK1
# RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn %t | FileCheck %s --check-prefixes=DISASM,DISASM1
# RUN: llvm-readelf -S -r %t.so | FileCheck %s --check-prefix=CHECK2
# RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn %t.so | FileCheck %s --check-prefixes=DISASM,DISASM2

# RUN: ld.lld %t.o %t2.so -z mark-plt -z now -o %t.mark
# RUN: llvm-readelf -S --dynamic-table -r %t.mark | FileCheck %s --check-prefix=MARK
# RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn %t.mark | FileCheck %s --check-prefix=DISASM-MARK

## --apply-dynamic-relocs must not trip the dynamic relocation addend check:
## the JUMP_SLOT addend is the PLT entry address while the .got.plt entry holds
## the lazy-binding address.
# RUN: ld.lld %t.o %t2.so -z mark-plt -z now --apply-dynamic-relocs -o %t.mark2
# RUN: llvm-readelf -r %t.mark2 | FileCheck %s --check-prefix=MARK-RELA

# MARK-RELA:      Relocation section '.rela.plt' at offset {{.*}} contains 2 entries:
# MARK-RELA:      {{.*}} R_X86_64_JUMP_SLOT 0000000000000000 weak + 2012c0
# MARK-RELA-NEXT: {{.*}} R_X86_64_JUMP_SLOT 0000000000000000 bar + 2012d0

## -z mark-plt requires RELA relocations to carry the PLT entry address addend.
# RUN: not ld.lld %t.o %t2.so -z mark-plt -z rel -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR-REL
# ERR-REL: error: -z mark-plt requires -z rela

# CHECK1:      Name      Type     Address          Off    Size   ES Flg Lk Inf Al
# CHECK1:      .plt      PROGBITS 00000000002012e0 0002e0 000030 00 AX   0   0 16
# CHECK1:      .got.plt  PROGBITS 00000000002033e0 0003e0 000028 00 WA   0   0  8
# CHECK1:      Relocation section '.rela.plt' at offset {{.*}} contains 2 entries:
# CHECK1:      00000000002033f8 {{.*}} R_X86_64_JUMP_SLOT 0000000000000000 weak + 0
# CHECK1-NEXT: 0000000000203400 {{.*}} R_X86_64_JUMP_SLOT 0000000000000000 bar + 0

# CHECK2:      Name      Type     Address          Off    Size   ES Flg Lk Inf Al
# CHECK2:      .plt      PROGBITS 0000000000001310 000310 000030 00 AX   0   0 16
# CHECK2:      .got.plt  PROGBITS 0000000000003400 000400 000028 00 WA   0   0  8
# CHECK2:      Relocation section '.rela.plt' at offset {{.*}} contains 2 entries:
# CHECK2:      0000000000003418 {{.*}} R_X86_64_JUMP_SLOT 0000000000000000 weak + 0
# CHECK2-NEXT: 0000000000003420 {{.*}} R_X86_64_JUMP_SLOT 0000000000000000 bar + 0

# MARK:      Name      Type     Address          Off    Size   ES Flg Lk Inf Al
# MARK:      .plt      PROGBITS 00000000002012b0 0002b0 000030 00 AX   0   0 16
# MARK:      0x0000000070000000 (X86_64_PLT)    0x2012b0
# MARK-NEXT: 0x0000000070000001 (X86_64_PLTSZ)  0x30
# MARK-NEXT: 0x0000000070000003 (X86_64_PLTENT) 0x10
# MARK:      Relocation section '.rela.plt' at offset {{.*}} contains 2 entries:
# MARK:      {{.*}} R_X86_64_JUMP_SLOT 0000000000000000 weak + 2012c0
# MARK-NEXT: {{.*}} R_X86_64_JUMP_SLOT 0000000000000000 bar + 2012d0

# DISASM:       <_start>:
# DISASM-NEXT:    callq {{.*}} <local>
# DISASM-NEXT:    callq {{.*}} <bar@plt>
# DISASM-NEXT:    jmp   {{.*}} <bar@plt>
# DISASM-NEXT:    jmp   {{.*}} <weak@plt>

# DISASM1:      Disassembly of section .plt:
# DISASM1-EMPTY:
# DISASM1-NEXT: <.plt>:
# DISASM1-NEXT: 2012e0:     pushq 8450(%rip)  # 0x2033e8
# DISASM1-NEXT:             jmpq *8452(%rip)  # 0x2033f0
# DISASM1-NEXT:             nopl (%rax)
# DISASM1-EMPTY:
# DISASM1-NEXT: <weak@plt>:
# DISASM1-NEXT: 2012f0:     jmpq *8450(%rip)  # 0x2033f8
# DISASM1-NEXT:             pushq $0
# DISASM1-NEXT:             jmp 0x2012e0 <.plt>
# DISASM1-EMPTY:
# DISASM1-NEXT: <bar@plt>:
# DISASM1-NEXT: 201300:     jmpq *8442(%rip)  # 0x203400
# DISASM1-NEXT:             pushq $1
# DISASM1-NEXT:             jmp 0x2012e0 <.plt>
# DISASM1-NOT:  {{.}}

# DISASM2:      Disassembly of section .plt:
# DISASM2-EMPTY:
# DISASM2-NEXT: <.plt>:
# DISASM2-NEXT:   1310:     pushq 8434(%rip)  # 0x3408
# DISASM2-NEXT:             jmpq *8436(%rip)  # 0x3410
# DISASM2-NEXT:             nopl (%rax)
# DISASM2-EMPTY:
# DISASM2-NEXT: <weak@plt>:
# DISASM2-NEXT:   1320:     jmpq *8434(%rip)  # 0x3418
# DISASM2-NEXT:             pushq $0
# DISASM2-NEXT:             jmp 0x1310 <.plt>
# DISASM2-EMPTY:
# DISASM2-NEXT: <bar@plt>:
# DISASM2-NEXT:   1330:     jmpq *8426(%rip)  # 0x3420
# DISASM2-NEXT:             pushq $1
# DISASM2-NEXT:             jmp 0x1310 <.plt>
# DISASM2-NOT:  {{.}}

# DISASM-MARK:      Disassembly of section .plt:
# DISASM-MARK-EMPTY:
# DISASM-MARK-NEXT: <.plt>:
# DISASM-MARK-NEXT: 2012b0:      pushq 4434(%rip)  # 0x202408
# DISASM-MARK-NEXT:              jmpq *4436(%rip)  # 0x202410
# DISASM-MARK-NEXT:              nopl (%rax)
# DISASM-MARK-EMPTY:
# DISASM-MARK: <weak@plt>:
# DISASM-MARK-NEXT: 2012c0:      jmpq *4434(%rip)  # 0x202418
# DISASM-MARK-NEXT:              pushq $0
# DISASM-MARK-NEXT:              jmp 0x2012b0 <.plt>
# DISASM-MARK-EMPTY:
# DISASM-MARK: <bar@plt>:
# DISASM-MARK-NEXT: 2012d0:      jmpq *4426(%rip)  # 0x202420
# DISASM-MARK-NEXT:              pushq $1
# DISASM-MARK-NEXT:              jmp 0x2012b0 <.plt>

.global _start
.weak weak

_start:
  call local
  call bar
  jmp bar@plt
  jmp weak

## foo is local and non-preemptale, no PLT is generated.
local:
  ret
