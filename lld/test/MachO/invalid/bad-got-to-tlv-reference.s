# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t.o
# RUN: not %lld -o /dev/null %t.o 2>&1 | FileCheck %s -DFILE=%t.o

# CHECK: error: [[FILE]]:(symbol _main+0x3): GOT_LOAD relocation requires that symbol _foo not be thread-local

.text
.globl _main
_main:
  movq _foo@GOTPCREL(%rip), %rax
  ret

.section __DATA,__thread_vars,thread_local_variables
_foo:
