# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t.o
# RUN: not lld -flavor darwinnew -o /dev/null %t.o 2>&1 | FileCheck %s

# CHECK: error: X86_64_RELOC_TLV must be used with movq instructions

.text
.globl _main
_main:
  leaq _foo@TLVP(%rip), %rax
  ret

.section	__DATA,__thread_vars,thread_local_variables
_foo:
