# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t.o
# RUN: not %lld -lSystem -init_offsets  %t.o -o /dev/null 2>&1 | FileCheck %s

# CHECK: error: {{.*}}init-offsets.s.tmp.o:(symbol _main+0x3): cannot reference _init_slot defined in __mod_init_func when -init_offsets is used

.globl _main
.text
_main:
  leaq _init_slot(%rip), %rax

.section __DATA,__mod_init_func,mod_init_funcs
_init_slot:
  .quad _main

