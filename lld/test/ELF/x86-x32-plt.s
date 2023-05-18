# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-gnux32 %s -o %t.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-gnux32 %p/Inputs/shared.s -o %t2.o
# RUN: ld.lld -shared -soname=t2 %t2.o -o %t2.so

# RUN: ld.lld %t.o %t2.so -o %t
# RUN: llvm-readelf -S -r %t | FileCheck %s
# RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn %t | FileCheck %s --check-prefixes=DISASM

# CHECK:      Name      Type     Address          Off    Size   ES Flg Lk Inf Al
# CHECK:      .plt      PROGBITS 002011e0 0001e0 000030 00 AX   0   0 16
# CHECK:      .got.plt  PROGBITS 00203278 000278 000028 00 WA   0   0  8
# CHECK:      Relocation section '.rela.plt' at offset {{.*}} contains 2 entries:
# CHECK:      00203290 {{.*}} R_X86_64_JUMP_SLOT 00000000 weak + 0
# CHECK-NEXT: 00203298 {{.*}} R_X86_64_JUMP_SLOT 00000000 bar + 0

# DISASM:       <_start>:
# DISASM-NEXT:    callq {{.*}} <local>
# DISASM-NEXT:    callq {{.*}} <bar@plt>
# DISASM-NEXT:    jmp   {{.*}} <bar@plt>
# DISASM-NEXT:    jmp   {{.*}} <weak@plt>

# DISASM:      Disassembly of section .plt:
# DISASM-EMPTY:
# DISASM-NEXT: <.plt>:
# DISASM-NEXT: 2011e0:     pushq 8346(%rip)  # 0x203280
# DISASM-NEXT:             jmpq *8348(%rip)  # 0x203288
# DISASM-NEXT:             nopl (%rax)
# DISASM-EMPTY:
# DISASM-NEXT: <weak@plt>:
# DISASM-NEXT: 2011f0:     jmpq *8346(%rip)  # 0x203290
# DISASM-NEXT:             pushq $0
# DISASM-NEXT:             jmp 0x2011e0 <.plt>
# DISASM-EMPTY:
# DISASM-NEXT: <bar@plt>:
# DISASM-NEXT: 201200:     jmpq *8338(%rip)  # 0x203298
# DISASM-NEXT:             pushq $1
# DISASM-NEXT:             jmp 0x2011e0 <.plt>
# DISASM-NOT:  {{.}}

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
