# REQUIRES: x86

## Allow local-dynamic R_X86_64_DTPOFF32 and R_X86_64_DTPOFF64 to preemptable symbols.

# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: ld.lld %t.o -shared -o %t.so
# RUN: llvm-objdump -d --no-show-raw-insn %t.so | FileCheck %s

# CHECK:      leaq    (%rax), %rax
# CHECK-NEXT: movabsq 0, %rax

## i is STB_GLOBAL and preemptable.
  leaq i@TLSLD(%rip), %rdi
  callq __tls_get_addr@PLT
  leaq i@DTPOFF(%rax), %rax # R_X86_64_DTPOFF32
  movabsq i@DTPOFF, %rax # R_X86_64_DTPOFF64

.section .tbss,"awT",@nobits
.globl i
i:
  .long 0
  .size i, 4
