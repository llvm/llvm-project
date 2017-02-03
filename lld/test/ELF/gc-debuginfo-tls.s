# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
# RUN: ld.lld %t.o --gc-sections -shared -o %t1
# RUN: ld.lld %t.o -shared -o %t2
# RUN: llvm-readobj -symbols %t1 | FileCheck %s --check-prefix=GC
# RUN: llvm-readobj -symbols %t2 | FileCheck %s --check-prefix=NOGC

# NOGC: Symbol {
# NOGC:   Name: (0)
# NOGC:   Value: 0x1000
# NOGC:   Size: 0
# NOGC:   Binding: Local
# NOGC:   Type: TLS
# NOGC:   Other: 0
# NOGC:   Section: .tbss
# NOGC: }

# GC-NOT: tbss

.section .tbss,"awT",@nobits
patatino:
  .long 0
  .section .noalloc,""
  .quad patatino
