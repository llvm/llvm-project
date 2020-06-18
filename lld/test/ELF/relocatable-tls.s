# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t1.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux \
# RUN:   %S/Inputs/relocatable-tls.s -o %t2.o

# RUN: ld.lld -r %t2.o -o %t3.r
# RUN: llvm-nm %t3.r | FileCheck %s
# CHECK: U __tls_get_addr

# RUN: ld.lld -shared %t2.o %t3.r -o %t4.out
# RUN: llvm-nm %t4.out | FileCheck %s

callq __tls_get_addr@PLT
