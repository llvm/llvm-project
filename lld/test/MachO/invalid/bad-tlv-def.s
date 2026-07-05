# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t.o
# RUN: not %lld -o /dev/null %t.o 2>&1 | FileCheck %s

# CHECK: error: GOT_LOAD relocation not allowed in thread-local section, must be UNSIGNED

.text
.globl _main
_main:
  ret

.section __DATA,__thread_vars,thread_local_variables
.globl _foo, _bar
_foo:
  movq _bar@GOTPCREL(%rip), %rax
