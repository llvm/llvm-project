# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: echo '.tbss; .globl tls; tls:' | llvm-mc -filetype=obj -triple=x86_64 - -o %tdef.o
# RUN: ld.lld %t.o -o - | llvm-objdump -d - | FileCheck %s

## A weak symbol does not fetch a lazy definition.
# RUN: ld.lld %t.o --start-lib %tdef.o --end-lib -o - | llvm-objdump -d - | FileCheck %s

## Undefined TLS symbols arbitrarily resolve to 0.
# CHECK:  leaq 16(%rax), %rdx

# RUN: ld.lld -shared %tdef.o -o %tdef.so
# RUN: not ld.lld %t.o %tdef.so -o /dev/null 2>&1 | FileCheck --check-prefix=COPYRELOC %s

# COPYRELOC: symbol 'tls' has no type

.weak tls
leaq tls@tpoff+16(%rax), %rdx
