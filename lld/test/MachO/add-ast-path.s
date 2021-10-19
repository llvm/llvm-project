# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t.o
# RUN: %lld -lSystem %t.o -o %t -add_ast_path asdf -add_ast_path fdsa
# RUN: dsymutil -s %t | FileCheck %s
# CHECK:      [     0] {{[0-9a-f]+}} 32 (N_AST        ) 00     0000   0000000000000000 'asdf'
# CHECK-NEXT: [     1] {{[0-9a-f]+}} 32 (N_AST        ) 00     0000   0000000000000000 'fdsa'

.globl _main

_main:
  ret
