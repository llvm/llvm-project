# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t.o
# RUN: not %lld -o %t %t.o %S/Inputs/macho-trie-node-loop 2>&1 | FileCheck %s
# CHECK: error:
# CHECK-SAME: /Inputs/macho-trie-node-loop: export trie child node infinite loop

.globl _main
_main:
  ret
