# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 %p/Inputs/shared.s -o %t2.o
# RUN: ld.lld -shared -soname=t2 %t2.o -o %t2.so

# RUN: ld.lld %t.o %t2.so -o %t
# RUN: ld.lld -shared %t.o %t2.so -o %t.so
# RUN: llvm-readelf -S -r %t | FileCheck %s --check-prefix=CHECK1
# RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck %s --check-prefixes=DISASM,DISASM1
# RUN: llvm-readelf -S -r %t.so | FileCheck %s --check-prefix=CHECK2
# RUN: llvm-objdump -d --no-show-raw-insn %t.so | FileCheck %s --check-prefixes=DISASM,DISASM2

# CHECK1:      Name      Type     Address          Off    Size   ES Flg Lk Inf Al
# CHECK1:      .plt      PROGBITS 00000000002012e0 0002e0 000030 00 AX   0   0 16
# CHECK1:      .got.plt  PROGBITS 00000000002033e0 0003e0 000028 00 WA   0   0  8
# CHECK1:      Relocation section '.rela.plt' at offset {{.*}} contains 2 entries:
# CHECK1:      00000000002033f8 {{.*}} R_X86_64_JUMP_SLOT 0000000000000000 bar + 0
# CHECK1-NEXT: 0000000000203400 {{.*}} R_X86_64_JUMP_SLOT 0000000000000000 weak + 0

# CHECK2:      Name      Type     Address          Off    Size   ES Flg Lk Inf Al
# CHECK2:      .plt      PROGBITS 0000000000001310 000310 000030 00 AX   0   0 16
# CHECK2:      .got.plt  PROGBITS 0000000000003400 000400 000028 00 WA   0   0  8
# CHECK2:      Relocation section '.rela.plt' at offset {{.*}} contains 2 entries:
# CHECK2:      0000000000003418 {{.*}} R_X86_64_JUMP_SLOT 0000000000000000 bar + 0
# CHECK2-NEXT: 0000000000003420 {{.*}} R_X86_64_JUMP_SLOT 0000000000000000 weak + 0

# DISASM:       <_start>:
# DISASM-NEXT:    callq {{.*}} <local>
# DISASM-NEXT:    callq {{.*}} <bar@plt>
# DISASM-NEXT:    jmp   {{.*}} <bar@plt>
# DISASM-NEXT:    jmp   {{.*}} <weak@plt>

# DISASM1:      Disassembly of section .plt:
# DISASM1-EMPTY:
# DISASM1-NEXT: <.plt>:
# DISASM1-NEXT: 2012e0:     pushq 8450(%rip)  # 2033e8
# DISASM1-NEXT:             jmpq *8452(%rip)  # 2033f0
# DISASM1-NEXT:             nopl (%rax)
# DISASM1-EMPTY:
# DISASM1-NEXT: <bar@plt>:
# DISASM1-NEXT: 2012f0:     jmpq *8450(%rip)  # 2033f8
# DISASM1-NEXT:             pushq $0
# DISASM1-NEXT:             jmp 0x2012e0 <.plt>
# DISASM1-EMPTY:
# DISASM1-NEXT: <weak@plt>:
# DISASM1-NEXT: 201300:     jmpq *8442(%rip)  # 203400
# DISASM1-NEXT:             pushq $1
# DISASM1-NEXT:             jmp 0x2012e0 <.plt>
# DISASM1-NOT:  {{.}}

# DISASM2:      Disassembly of section .plt:
# DISASM2-EMPTY:
# DISASM2-NEXT: <.plt>:
# DISASM2-NEXT:   1310:     pushq 8434(%rip)  # 3408
# DISASM2-NEXT:             jmpq *8436(%rip)  # 3410
# DISASM2-NEXT:             nopl (%rax)
# DISASM2-EMPTY:
# DISASM2-NEXT: <bar@plt>:
# DISASM2-NEXT:   1320:     jmpq *8434(%rip)  # 3418
# DISASM2-NEXT:             pushq $0
# DISASM2-NEXT:             jmp 0x1310 <.plt>
# DISASM2-EMPTY:
# DISASM2-NEXT: <weak@plt>:
# DISASM2-NEXT:   1330:     jmpq *8426(%rip)  # 3420
# DISASM2-NEXT:             pushq $1
# DISASM2-NEXT:             jmp 0x1310 <.plt>
# DISASM2-NOT:  {{.}}

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
