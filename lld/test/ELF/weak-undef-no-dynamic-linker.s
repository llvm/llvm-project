# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: ld.lld -pie %t.o -o %t
# RUN: llvm-readobj --dyn-syms %t | FileCheck --check-prefix=NO %s
# RUN: ld.lld -pie --no-dynamic-linker %t.o -o %t
# RUN: llvm-readobj --dyn-syms %t | FileCheck --check-prefix=NO %s

## With static PIE (whether or not --no-dynamic-linker is specified), don't
## emit undefined weak symbols to .dynsym . This suppresses relocations.
# NO-NOT: Name: foo

.weak foo
cmpq $0, foo@GOTPCREL(%rip)
callq foo
