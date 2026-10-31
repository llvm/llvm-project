# REQUIRES: x86
## ILP32 code materializes the TLS offset with a 32-bit operand, so GOTTPOFF
## lands on movl/addl rather than the REX.W movq/addq forms.

# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux-gnux32 %s -o %t.o
# RUN: ld.lld -m elf32_x86_64 %t.o -o %t1
# RUN: llvm-readobj -r %t1 | FileCheck --check-prefix=NORELOC %s
# RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn %t1 | FileCheck --check-prefix=DISASM %s

# NORELOC:      Relocations [
# NORELOC-NEXT: ]

# DISASM:      <_start>:
# DISASM-NEXT:   movl $-4, %eax
# DISASM-NEXT:   leal -4(%rcx), %ecx
## ADD is used instead of LEA for ESP, which would need a SIB byte.
# DISASM-NEXT:   addl $-4, %esp
# DISASM-NEXT:   movl $-4, %r8d
# DISASM-NEXT:   leal -4(%r9), %r9d

.type tls0,@object
.section .tbss,"awT",@nobits
.globl tls0
.align 4
tls0:
 .long 0
 .size tls0, 4

.text
.globl _start
_start:
  movl 0(%rip), %eax
  .reloc .-4, R_X86_64_GOTTPOFF, tls0-4

  addl 0(%rip), %ecx
  .reloc .-4, R_X86_64_GOTTPOFF, tls0-4

  addl 0(%rip), %esp
  .reloc .-4, R_X86_64_GOTTPOFF, tls0-4

  movl 0(%rip), %r8d
  .reloc .-4, R_X86_64_GOTTPOFF, tls0-4

  addl 0(%rip), %r9d
  .reloc .-4, R_X86_64_GOTTPOFF, tls0-4
