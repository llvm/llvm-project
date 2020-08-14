# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t.o
# RUN: not lld -flavor darwinnew -o /dev/null %t.o 2>&1 | FileCheck %s

# CHECK: error: relocations in thread-local variable sections must be X86_64_RELOC_UNSIGNED

.text
.globl _main
_main:
  ret

.section	__DATA,__thread_vars,thread_local_variables
.globl	_foo, _bar
_foo:
  movq _bar@GOTPCREL(%rip), %rax
