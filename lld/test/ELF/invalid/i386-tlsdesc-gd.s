# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=i386 %s -o %t.o
# RUN: echo '.tbss; .globl a; a:' | llvm-mc -filetype=obj -triple=i386 - -o %t1.o
# RUN: ld.lld -shared %t1.o -o %t1.so

## GD to LE relaxation.
# RUN: not ld.lld %t.o %t1.o -o /dev/null 2>&1 | FileCheck -DINPUT=%t.o %s
## GD to IE relaxation.
# RUN: not ld.lld %t.o %t1.so -o /dev/null 2>&1 | FileCheck -DINPUT=%t.o %s

# CHECK: error: [[INPUT]]:(.text+0x0): R_386_TLS_GOTDESC must be used in leal x@tlsdesc(%ebx), %eax

leal a@tlsdesc(%ebx), %ecx
call *a@tlscall(%eax)
