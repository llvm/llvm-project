# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: ld.lld -shared %t.o -o %t.so
# RUN: llvm-readobj -r %t.so | FileCheck %s --check-prefix=RELA

## Both TLSDESC and DTPMOD64/DTPOFF64 should be present.
# RELA:      .rela.dyn {
# RELA-NEXT:   0x[[#%X,ADDR:]] R_X86_64_TLSDESC  a 0x0
# RELA-NEXT:   0x[[#ADDR+16]]  R_X86_64_DTPMOD64 a 0x0
# RELA-NEXT:   0x[[#ADDR+24]]  R_X86_64_DTPOFF64 a 0x0
# RELA-NEXT: }

leaq a@tlsdesc(%rip), %rax
call *a@tlscall(%rax)
movl %fs:(%rax), %eax

.byte 0x66
leaq a@tlsgd(%rip), %rdi
.word 0x6666
rex64
call __tls_get_addr@PLT

.section .tbss
.globl a
.zero 8
a:
.zero 4
