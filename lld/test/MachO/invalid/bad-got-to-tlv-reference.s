# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t.o
# RUN: not lld -flavor darwinnew -o /dev/null %t.o 2>&1 | FileCheck %s -DFILE=%t.o

# CHECK: error: found GOT relocation referencing thread-local variable in [[FILE]]:(__text)

.text
.globl _main
_main:
  movq _foo@GOTPCREL(%rip), %rax
  ret

.section __DATA,__thread_vars,thread_local_variables
_foo:
